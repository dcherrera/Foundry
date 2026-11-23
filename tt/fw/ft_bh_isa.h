/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_bh_isa.h
 * @brief Blackhole Tensix coprocessor instruction encodings: the 32-bit word for each instruction
 *   the kernels issue, the two issue paths (.ttinsn immediate and instruction-buffer store), and
 *   the field-value constants those words take.
 *
 * Source: tt-isa-documentation. Each instruction cites the page it was read from (BlackholeA0
 * where a Blackhole page exists, else WormholeB0) and the Diagrams/Src/Bits32.lua layout entry
 * (the `_BH` entry where one exists). Bit positions not drawn by any Bits32 entry are marked
 * "inferred" with the reasoning; those are the fields to check first against the instruction
 * diff.
 *
 * Every FT_BH_ENC_<I> is a constant expression when its arguments are, so FT_BH_TTINSN_<I> can
 * be used with compile-time arguments and FT_BH_PUSH_<I> with run-time ones.
 *
 * @status Clean-room replacement of the kernels' instruction-encoding header, plan Phase 4.2.
 * @issues Blackhole deltas measured by the instruction diff: AddrMod is 3 bits at 16:14 for
 *   ELW*, MVMUL and ZEROACC (MOVD2B follows, unexercised) but stays 2 bits at 16:15 for GAPOOL
 *   and GMPOOL; ZEROACC is relaid. UNPACR bits 5 and 0 and UNPACR_NOP bit 5 are inferred
 *   (see each).
 * @todo None
 */
#ifndef FT_BH_ISA_H
#define FT_BH_ISA_H

#include <stdint.h>

#include "ft_bh_regs.h"

/* ===== Issue mechanisms ================================================================== */

/* Opcode lives in bits 31:24 of every Tensix instruction word. */
#define FT_BH_ENC(opcode, fields) (((opcode) << 24) + (fields))

/* Immediate issue: the .ttinsn directive (BabyRISCV/PushTensixInstruction.md). */
#define FT_BH_TTINSN(word) __asm__ __volatile__(".ttinsn %0" : : "n"((word)))

/* Store issue: the instruction buffer at INSTRN_BUF_BASE 0xFFE40000, placed by the linker script
 * (BabyRISCV/PushTensixInstruction.md). */
extern volatile uint32_t __instrn_buffer[];
#define FT_BH_INSTRN_BUF ((volatile uint32_t FT_BH_REG_PTR *)__instrn_buffer)
#define FT_BH_PUSH(word) (FT_BH_INSTRN_BUF[0] = (word))

/* ===== Zero-operand instructions ========================================================= */

/* NOP.md (WH page), Bits32 NOP. */
#define FT_BH_ENC_NOP FT_BH_ENC(0x02u, 0)
#define FT_BH_TTINSN_NOP FT_BH_TTINSN(FT_BH_ENC_NOP)
#define FT_BH_PUSH_NOP FT_BH_PUSH(FT_BH_ENC_NOP)

/* DMANOP.md (WH page), Bits32 DMANOP. */
#define FT_BH_ENC_DMANOP FT_BH_ENC(0x60u, 0)
#define FT_BH_TTINSN_DMANOP FT_BH_TTINSN(FT_BH_ENC_DMANOP)
#define FT_BH_PUSH_DMANOP FT_BH_PUSH(FT_BH_ENC_DMANOP)

/* TRNSPSRCB.md (WH page), Bits32 TRNSPSRCB. */
#define FT_BH_ENC_TRNSPSRCB FT_BH_ENC(0x16u, 0)
#define FT_BH_TTINSN_TRNSPSRCB FT_BH_TTINSN(FT_BH_ENC_TRNSPSRCB)
#define FT_BH_PUSH_TRNSPSRCB FT_BH_PUSH(FT_BH_ENC_TRNSPSRCB)

/* SFPNOP.md (BH page defers to WH, identical), Bits32 SFPNOP. */
#define FT_BH_ENC_SFPNOP FT_BH_ENC(0x8Fu, 0)
#define FT_BH_TTINSN_SFPNOP FT_BH_TTINSN(FT_BH_ENC_SFPNOP)
#define FT_BH_PUSH_SFPNOP FT_BH_PUSH(FT_BH_ENC_SFPNOP)

/* ===== Matrix Unit (FPU) ================================================================= */
/* No Blackhole pages exist for these (BH MatrixUnit.md defers to WH); WH layouts used except
 * AddrMod: WH draws 2 bits at 16:15, Blackhole uses 3 bits at 16:14 (Blackhole layout, measured
 * by the instruction diff: ELWADD/ELWMUL/MVMUL words; MOVD2B follows, unexercised). GAPOOL and
 * GMPOOL are the exception: they keep the WH 2-bit AddrMod at 16:15 (measured). */

/* ELWADD.md / ELWSUB.md / ELWMUL.md (WH pages), Bits32 ELWADD / ELWSUB / ELWMUL.
 * clear_dvalid: FlipSrcB<<1 | FlipSrcA at 23:22. add_to_dest: AddDst at 21 (ELWMUL draws no
 * AddDst bit; same position kept). srcb_bcast: BroadcastSrcBRow<<1 | BroadcastSrcBCol0 at 20:19.
 * addr_mod: AddrMod at 16:14 (Blackhole layout, measured by the instruction diff). dst_row: DstRow
 * at 9:0. */
#define FT_BH_ELW_FIELDS(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)              \
    (((clear_dvalid) << 22) + ((add_to_dest) << 21) + ((srcb_bcast) << 19) + ((addr_mod) << 14) \
     + ((dst_row) << 0))

#define FT_BH_ENC_ELWADD(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)               \
    FT_BH_ENC(0x28u, FT_BH_ELW_FIELDS(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))
#define FT_BH_TTINSN_ELWADD(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)            \
    FT_BH_TTINSN(FT_BH_ENC_ELWADD(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))
#define FT_BH_PUSH_ELWADD(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)              \
    FT_BH_PUSH(FT_BH_ENC_ELWADD(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))

#define FT_BH_ENC_ELWSUB(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)               \
    FT_BH_ENC(0x30u, FT_BH_ELW_FIELDS(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))
#define FT_BH_TTINSN_ELWSUB(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)            \
    FT_BH_TTINSN(FT_BH_ENC_ELWSUB(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))
#define FT_BH_PUSH_ELWSUB(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)              \
    FT_BH_PUSH(FT_BH_ENC_ELWSUB(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))

#define FT_BH_ENC_ELWMUL(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)               \
    FT_BH_ENC(0x27u, FT_BH_ELW_FIELDS(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))
#define FT_BH_TTINSN_ELWMUL(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)            \
    FT_BH_TTINSN(FT_BH_ENC_ELWMUL(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))
#define FT_BH_PUSH_ELWMUL(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row)              \
    FT_BH_PUSH(FT_BH_ENC_ELWMUL(clear_dvalid, add_to_dest, srcb_bcast, addr_mod, dst_row))

/* MVMUL.md (WH page), Bits32 MVMUL. clear_dvalid at 23:22; mode: BroadcastSrcBRow at 19 (the
 * field base); addr_mod at 16:14 (Blackhole layout, measured by the instruction diff); dst_row
 * at 9:0. */
#define FT_BH_ENC_MVMUL(clear_dvalid, mode, addr_mod, dst_row)                                   \
    FT_BH_ENC(0x26u, ((clear_dvalid) << 22) + ((mode) << 19) + ((addr_mod) << 14)              \
                         + ((dst_row) << 0))
#define FT_BH_TTINSN_MVMUL(clear_dvalid, mode, addr_mod, dst_row)                                \
    FT_BH_TTINSN(FT_BH_ENC_MVMUL(clear_dvalid, mode, addr_mod, dst_row))
#define FT_BH_PUSH_MVMUL(clear_dvalid, mode, addr_mod, dst_row)                                  \
    FT_BH_PUSH(FT_BH_ENC_MVMUL(clear_dvalid, mode, addr_mod, dst_row))

/* GAPOOL.md / GMPOOL.md (WH pages), Bits32 GAPOOL / GMPOOL. clear_dvalid at 23:22; addr_mod at
 * 16:15, 2 bits, and max_pool_index (ArgMax) at 14: Blackhole layout, measured by the instruction
 * diff (GAPOOL store-form word 0x34088xxx, round 2), i.e. the WH layout, unlike the 3-bit AddrMod
 * of ELW* and MVMUL. dst_row at 9:0. mode: no Bits32 entry draws it (GMPOOL's syntax passes a fixed
 * 1 there); placed at 19, inferred from MVMUL/ELW* and consistent with bit 19 set in the measured
 * word. */
#define FT_BH_POOL_FIELDS(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)                \
    (((clear_dvalid) << 22) + ((mode) << 19) + ((addr_mod) << 15) + ((max_pool_index) << 14)   \
     + ((dst_row) << 0))

#define FT_BH_ENC_GAPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)                  \
    FT_BH_ENC(0x34u, FT_BH_POOL_FIELDS(clear_dvalid, mode, addr_mod, max_pool_index, dst_row))
#define FT_BH_TTINSN_GAPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)               \
    FT_BH_TTINSN(FT_BH_ENC_GAPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row))
#define FT_BH_PUSH_GAPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)                 \
    FT_BH_PUSH(FT_BH_ENC_GAPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row))

#define FT_BH_ENC_GMPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)                  \
    FT_BH_ENC(0x33u, FT_BH_POOL_FIELDS(clear_dvalid, mode, addr_mod, max_pool_index, dst_row))
#define FT_BH_TTINSN_GMPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)               \
    FT_BH_TTINSN(FT_BH_ENC_GMPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row))
#define FT_BH_PUSH_GMPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row)                 \
    FT_BH_PUSH(FT_BH_ENC_GMPOOL(clear_dvalid, mode, addr_mod, max_pool_index, dst_row))

/* MOVD2B.md (WH page), Bits32 MOVD2B. use_dst32b_lo: UseDst32bLo at 23; src_row: SrcRow at
 * 22:17; addr_mod at 16:14 (Blackhole Matrix Unit layout, measured by the instruction diff on
 * ELWADD/ELWMUL/MVMUL; not exercised for MOVD2B); mode: field base 12 (Move4Rows is mode bit 1,
 * bit 13); dst_row at 9:0. */
#define FT_BH_ENC_MOVD2B(use_dst32b_lo, src_row, addr_mod, mode, dst_row)                        \
    FT_BH_ENC(0x0Au, ((use_dst32b_lo) << 23) + ((src_row) << 17) + ((addr_mod) << 14)          \
                         + ((mode) << 12) + ((dst_row) << 0))
#define FT_BH_TTINSN_MOVD2B(use_dst32b_lo, src_row, addr_mod, mode, dst_row)                     \
    FT_BH_TTINSN(FT_BH_ENC_MOVD2B(use_dst32b_lo, src_row, addr_mod, mode, dst_row))
#define FT_BH_PUSH_MOVD2B(use_dst32b_lo, src_row, addr_mod, mode, dst_row)                       \
    FT_BH_PUSH(FT_BH_ENC_MOVD2B(use_dst32b_lo, src_row, addr_mod, mode, dst_row))

/* ZEROACC.md (WH page), Bits32 ZEROACC. Blackhole layout, measured by the instruction diff
 * (WH draws UseDst32b 21, Mode 20:19, Revert 18, AddrMod 16:15): clear_mode: Mode at 20:19;
 * use_dst32b at 18; clear_zero_flags (Revert) at 17 (reported with the diff; not exercised by a
 * non-zero value); addr_mod at 16:14; where: Imm10 at 9:0. */
#define FT_BH_ENC_ZEROACC(clear_mode, use_dst32b, clear_zero_flags, addr_mod, where)             \
    FT_BH_ENC(0x10u, ((clear_mode) << 19) + ((use_dst32b) << 18) + ((clear_zero_flags) << 17)  \
                         + ((addr_mod) << 14) + ((where) << 0))
#define FT_BH_TTINSN_ZEROACC(clear_mode, use_dst32b, clear_zero_flags, addr_mod, where)          \
    FT_BH_TTINSN(FT_BH_ENC_ZEROACC(clear_mode, use_dst32b, clear_zero_flags, addr_mod, where))
#define FT_BH_PUSH_ZEROACC(clear_mode, use_dst32b, clear_zero_flags, addr_mod, where)            \
    FT_BH_PUSH(FT_BH_ENC_ZEROACC(clear_mode, use_dst32b, clear_zero_flags, addr_mod, where))

/* ZEROSRC.md (WH page), Bits32 ZEROSRC. zero_value: NegativeInfSrcA at 4; write_mode:
 * SingleBankMatrixUnit at 3; bank_mask: BothBanks at 2; src_mask: ClearSrcB<<1 | ClearSrcA at
 * 1:0. */
#define FT_BH_ENC_ZEROSRC(zero_value, write_mode, bank_mask, src_mask)                           \
    FT_BH_ENC(0x11u, ((zero_value) << 4) + ((write_mode) << 3) + ((bank_mask) << 2)            \
                         + ((src_mask) << 0))
#define FT_BH_TTINSN_ZEROSRC(zero_value, write_mode, bank_mask, src_mask)                        \
    FT_BH_TTINSN(FT_BH_ENC_ZEROSRC(zero_value, write_mode, bank_mask, src_mask))
#define FT_BH_PUSH_ZEROSRC(zero_value, write_mode, bank_mask, src_mask)                          \
    FT_BH_PUSH(FT_BH_ENC_ZEROSRC(zero_value, write_mode, bank_mask, src_mask))

/* ===== Register-window counters ========================================================== */

/* INCRWC.md (WH page), Bits32 INCRWC. cr: DstCr<<2 | SrcBCr<<1 | SrcACr at 20:18; dst_inc at
 * 17:14; srcb_inc at 13:10; srca_inc at 9:6. */
#define FT_BH_ENC_INCRWC(cr, dst_inc, srcb_inc, srca_inc)                                        \
    FT_BH_ENC(0x38u, ((cr) << 18) + ((dst_inc) << 14) + ((srcb_inc) << 10) + ((srca_inc) << 6))
#define FT_BH_TTINSN_INCRWC(cr, dst_inc, srcb_inc, srca_inc)                                     \
    FT_BH_TTINSN(FT_BH_ENC_INCRWC(cr, dst_inc, srcb_inc, srca_inc))
#define FT_BH_PUSH_INCRWC(cr, dst_inc, srcb_inc, srca_inc)                                       \
    FT_BH_PUSH(FT_BH_ENC_INCRWC(cr, dst_inc, srcb_inc, srca_inc))

/* SETRWC.md (WH page), Bits32 SETRWC. clear_dvalid: FlipSrcB<<1 | FlipSrcA at 23:22; cr:
 * DstCtoCr<<3 | DstCr<<2 | SrcBCr<<1 | SrcACr at 21:18; dst at 17:14; srcb at 13:10; srca at 9:6;
 * set_mask: Fidelity<<3 | Dst<<2 | SrcB<<1 | SrcA at 3:0. */
#define FT_BH_ENC_SETRWC(clear_dvalid, cr, dst, srcb, srca, set_mask)                            \
    FT_BH_ENC(0x37u, ((clear_dvalid) << 22) + ((cr) << 18) + ((dst) << 14) + ((srcb) << 10)    \
                         + ((srca) << 6) + ((set_mask) << 0))
#define FT_BH_TTINSN_SETRWC(clear_dvalid, cr, dst, srcb, srca, set_mask)                         \
    FT_BH_TTINSN(FT_BH_ENC_SETRWC(clear_dvalid, cr, dst, srcb, srca, set_mask))
#define FT_BH_PUSH_SETRWC(clear_dvalid, cr, dst, srcb, srca, set_mask)                           \
    FT_BH_PUSH(FT_BH_ENC_SETRWC(clear_dvalid, cr, dst, srcb, srca, set_mask))

/* ===== Configuration Unit ================================================================ */

/* SETC16.md (BH page), Bits32 SETC16. cfg_index: CfgIndex at 23:16; value: NewValue at 15:0. */
#define FT_BH_ENC_SETC16(cfg_index, value)                                                       \
    FT_BH_ENC(0xB2u, ((cfg_index) << 16) + ((value) << 0))
#define FT_BH_TTINSN_SETC16(cfg_index, value) FT_BH_TTINSN(FT_BH_ENC_SETC16(cfg_index, value))
#define FT_BH_PUSH_SETC16(cfg_index, value) FT_BH_PUSH(FT_BH_ENC_SETC16(cfg_index, value))

/* RMWCIB.md (BH page), Bits32 RMWCIB. Opcode 0xB3 + byte lane. mask: Mask at 23:16; data:
 * NewValue at 15:8; cfg_word: Index4 at 7:0. */
#define FT_BH_RMWCIB_FIELDS(mask, data, cfg_word)                                               \
    (((mask) << 16) + ((data) << 8) + ((cfg_word) << 0))

#define FT_BH_ENC_RMWCIB0(mask, data, cfg_word)                                                  \
    FT_BH_ENC(0xB3u, FT_BH_RMWCIB_FIELDS(mask, data, cfg_word))
#define FT_BH_TTINSN_RMWCIB0(mask, data, cfg_word)                                               \
    FT_BH_TTINSN(FT_BH_ENC_RMWCIB0(mask, data, cfg_word))
#define FT_BH_PUSH_RMWCIB0(mask, data, cfg_word) FT_BH_PUSH(FT_BH_ENC_RMWCIB0(mask, data, cfg_word))

#define FT_BH_ENC_RMWCIB1(mask, data, cfg_word)                                                  \
    FT_BH_ENC(0xB4u, FT_BH_RMWCIB_FIELDS(mask, data, cfg_word))
#define FT_BH_TTINSN_RMWCIB1(mask, data, cfg_word)                                               \
    FT_BH_TTINSN(FT_BH_ENC_RMWCIB1(mask, data, cfg_word))
#define FT_BH_PUSH_RMWCIB1(mask, data, cfg_word) FT_BH_PUSH(FT_BH_ENC_RMWCIB1(mask, data, cfg_word))

#define FT_BH_ENC_RMWCIB2(mask, data, cfg_word)                                                  \
    FT_BH_ENC(0xB5u, FT_BH_RMWCIB_FIELDS(mask, data, cfg_word))
#define FT_BH_TTINSN_RMWCIB2(mask, data, cfg_word)                                               \
    FT_BH_TTINSN(FT_BH_ENC_RMWCIB2(mask, data, cfg_word))
#define FT_BH_PUSH_RMWCIB2(mask, data, cfg_word) FT_BH_PUSH(FT_BH_ENC_RMWCIB2(mask, data, cfg_word))

#define FT_BH_ENC_RMWCIB3(mask, data, cfg_word)                                                  \
    FT_BH_ENC(0xB6u, FT_BH_RMWCIB_FIELDS(mask, data, cfg_word))
#define FT_BH_TTINSN_RMWCIB3(mask, data, cfg_word)                                               \
    FT_BH_TTINSN(FT_BH_ENC_RMWCIB3(mask, data, cfg_word))
#define FT_BH_PUSH_RMWCIB3(mask, data, cfg_word) FT_BH_PUSH(FT_BH_ENC_RMWCIB3(mask, data, cfg_word))

/* WRCFG.md (BH page defers to WH, identical), Bits32 WRCFG. gpr: InputReg at 21:16; is_128bit:
 * Is128Bit at 15; cfg_word: CfgIndex at 10:0. */
#define FT_BH_ENC_WRCFG(gpr, is_128bit, cfg_word)                                                \
    FT_BH_ENC(0xB0u, ((gpr) << 16) + ((is_128bit) << 15) + ((cfg_word) << 0))
#define FT_BH_TTINSN_WRCFG(gpr, is_128bit, cfg_word)                                             \
    FT_BH_TTINSN(FT_BH_ENC_WRCFG(gpr, is_128bit, cfg_word))
#define FT_BH_PUSH_WRCFG(gpr, is_128bit, cfg_word)                                               \
    FT_BH_PUSH(FT_BH_ENC_WRCFG(gpr, is_128bit, cfg_word))

/* RDCFG.md (BH page), Bits32 RDCFG. gpr: ResultReg at 21:16; cfg_word: CfgIndex at 10:0. */
#define FT_BH_ENC_RDCFG(gpr, cfg_word) FT_BH_ENC(0xB1u, ((gpr) << 16) + ((cfg_word) << 0))
#define FT_BH_TTINSN_RDCFG(gpr, cfg_word) FT_BH_TTINSN(FT_BH_ENC_RDCFG(gpr, cfg_word))
#define FT_BH_PUSH_RDCFG(gpr, cfg_word) FT_BH_PUSH(FT_BH_ENC_RDCFG(gpr, cfg_word))

/* ===== Sync Unit ========================================================================= */

/* STALLWAIT.md (BH page), Bits32 STALLWAIT_BH. stall_on: BlockMask at 23:15; wait_for:
 * ConditionMask at 12:0 (13 bits on BH; WH is 15 bits). */
#define FT_BH_ENC_STALLWAIT(stall_on, wait_for)                                                  \
    FT_BH_ENC(0xA2u, ((stall_on) << 15) + ((wait_for) << 0))
#define FT_BH_TTINSN_STALLWAIT(stall_on, wait_for)                                               \
    FT_BH_TTINSN(FT_BH_ENC_STALLWAIT(stall_on, wait_for))
#define FT_BH_PUSH_STALLWAIT(stall_on, wait_for) FT_BH_PUSH(FT_BH_ENC_STALLWAIT(stall_on, wait_for))

/* SEMINIT.md (BH page defers to WH, identical), Bits32 SEMINIT. max_value: NewMax at 23:20;
 * init_value: NewValue at 19:16; sem_mask: SemaphoreMask at 9:2. */
#define FT_BH_ENC_SEMINIT(max_value, init_value, sem_mask)                                       \
    FT_BH_ENC(0xA3u, ((max_value) << 20) + ((init_value) << 16) + ((sem_mask) << 2))
#define FT_BH_TTINSN_SEMINIT(max_value, init_value, sem_mask)                                    \
    FT_BH_TTINSN(FT_BH_ENC_SEMINIT(max_value, init_value, sem_mask))
#define FT_BH_PUSH_SEMINIT(max_value, init_value, sem_mask)                                      \
    FT_BH_PUSH(FT_BH_ENC_SEMINIT(max_value, init_value, sem_mask))

/* SEMPOST.md / SEMGET.md (BH pages defer to WH, identical), Bits32 SEMPOST / SEMGET.
 * sem_mask: SemaphoreMask at 9:2. */
#define FT_BH_ENC_SEMPOST(sem_mask) FT_BH_ENC(0xA4u, ((sem_mask) << 2))
#define FT_BH_TTINSN_SEMPOST(sem_mask) FT_BH_TTINSN(FT_BH_ENC_SEMPOST(sem_mask))
#define FT_BH_PUSH_SEMPOST(sem_mask) FT_BH_PUSH(FT_BH_ENC_SEMPOST(sem_mask))

#define FT_BH_ENC_SEMGET(sem_mask) FT_BH_ENC(0xA5u, ((sem_mask) << 2))
#define FT_BH_TTINSN_SEMGET(sem_mask) FT_BH_TTINSN(FT_BH_ENC_SEMGET(sem_mask))
#define FT_BH_PUSH_SEMGET(sem_mask) FT_BH_PUSH(FT_BH_ENC_SEMGET(sem_mask))

/* SEMWAIT.md (BH page), Bits32 SEMWAIT. stall_on: BlockMask at 23:15; sem_mask: SemaphoreMask
 * at 9:2; wait_cond: ConditionMask at 1:0. */
#define FT_BH_ENC_SEMWAIT(stall_on, sem_mask, wait_cond)                                         \
    FT_BH_ENC(0xA6u, ((stall_on) << 15) + ((sem_mask) << 2) + ((wait_cond) << 0))
#define FT_BH_TTINSN_SEMWAIT(stall_on, sem_mask, wait_cond)                                      \
    FT_BH_TTINSN(FT_BH_ENC_SEMWAIT(stall_on, sem_mask, wait_cond))
#define FT_BH_PUSH_SEMWAIT(stall_on, sem_mask, wait_cond)                                        \
    FT_BH_PUSH(FT_BH_ENC_SEMWAIT(stall_on, sem_mask, wait_cond))

/* ATGETM.md / ATRELM.md (BH pages), Bits32 ATGETM / ATRELM. mutex: Index at 15:0. */
#define FT_BH_ENC_ATGETM(mutex) FT_BH_ENC(0xA0u, ((mutex) << 0))
#define FT_BH_TTINSN_ATGETM(mutex) FT_BH_TTINSN(FT_BH_ENC_ATGETM(mutex))
#define FT_BH_PUSH_ATGETM(mutex) FT_BH_PUSH(FT_BH_ENC_ATGETM(mutex))

#define FT_BH_ENC_ATRELM(mutex) FT_BH_ENC(0xA1u, ((mutex) << 0))
#define FT_BH_TTINSN_ATRELM(mutex) FT_BH_TTINSN(FT_BH_ENC_ATRELM(mutex))
#define FT_BH_PUSH_ATRELM(mutex) FT_BH_PUSH(FT_BH_ENC_ATRELM(mutex))

/* ===== Frontend: MOP expander and replay buffer ========================================== */

/* MOP.md (WH page; BH MOPExpander.md has no MOP encoding), Bits32 MOP. mop_type: Template at 23;
 * loop_count: Count1 at 22:16; zmask_lo: MaskLo at 15:0. */
#define FT_BH_ENC_MOP(mop_type, loop_count, zmask_lo)                                            \
    FT_BH_ENC(0x01u, ((mop_type) << 23) + ((loop_count) << 16) + ((zmask_lo) << 0))
#define FT_BH_TTINSN_MOP(mop_type, loop_count, zmask_lo)                                         \
    FT_BH_TTINSN(FT_BH_ENC_MOP(mop_type, loop_count, zmask_lo))
#define FT_BH_PUSH_MOP(mop_type, loop_count, zmask_lo)                                           \
    FT_BH_PUSH(FT_BH_ENC_MOP(mop_type, loop_count, zmask_lo))

/* REPLAY.md (BH page defers to WH, "similar"), Bits32 REPLAY (no _BH entry). start: Index at
 * 18:14; length: Count at 9:4; exec_while_loading: Exec at 1; load_mode: Load at 0. */
#define FT_BH_ENC_REPLAY(start, length, exec_while_loading, load_mode)                           \
    FT_BH_ENC(0x04u, ((start) << 14) + ((length) << 4) + ((exec_while_loading) << 1)           \
                         + ((load_mode) << 0))
#define FT_BH_TTINSN_REPLAY(start, length, exec_while_loading, load_mode)                        \
    FT_BH_TTINSN(FT_BH_ENC_REPLAY(start, length, exec_while_loading, load_mode))
#define FT_BH_PUSH_REPLAY(start, length, exec_while_loading, load_mode)                          \
    FT_BH_PUSH(FT_BH_ENC_REPLAY(start, length, exec_while_loading, load_mode))

/* ===== Scalar Unit (ThCon) =============================================================== */

/* SETDMAREG_Immediate.md / SETDMAREG_Special.md (WH pages), Bits32 SETDMAREG_Immediate /
 * SETDMAREG_Special. sig_size: ResultSize at 23:22; payload: NewValue at 23:8 (or the special
 * selectors WhichPackers/InputSource/InputHalfReg starting at 8); set_signals_mode: bit 7;
 * reg_index: ResultHalfReg at 6:0. */
#define FT_BH_ENC_SETDMAREG(sig_size, payload, set_signals_mode, reg_index)                      \
    FT_BH_ENC(0x45u, ((sig_size) << 22) + ((payload) << 8) + ((set_signals_mode) << 7)         \
                         + ((reg_index) << 0))
#define FT_BH_TTINSN_SETDMAREG(sig_size, payload, set_signals_mode, reg_index)                   \
    FT_BH_TTINSN(FT_BH_ENC_SETDMAREG(sig_size, payload, set_signals_mode, reg_index))
#define FT_BH_PUSH_SETDMAREG(sig_size, payload, set_signals_mode, reg_index)                     \
    FT_BH_PUSH(FT_BH_ENC_SETDMAREG(sig_size, payload, set_signals_mode, reg_index))

/* ADDDMAREG.md (WH page), Bits32 ADDDMAREG / ADDDMAREGi. b_is_const: bit 23; result: ResultReg
 * at 17:12; operand_b: RightReg or RightImm6 at 11:6; operand_a: LeftReg at 5:0. */
#define FT_BH_ENC_ADDDMAREG(b_is_const, result, operand_b, operand_a)                            \
    FT_BH_ENC(0x58u, ((b_is_const) << 23) + ((result) << 12) + ((operand_b) << 6)              \
                         + ((operand_a) << 0))
#define FT_BH_TTINSN_ADDDMAREG(b_is_const, result, operand_b, operand_a)                         \
    FT_BH_TTINSN(FT_BH_ENC_ADDDMAREG(b_is_const, result, operand_b, operand_a))
#define FT_BH_PUSH_ADDDMAREG(b_is_const, result, operand_b, operand_a)                           \
    FT_BH_PUSH(FT_BH_ENC_ADDDMAREG(b_is_const, result, operand_b, operand_a))

/* STOREREG.md (WH page), Bits32 STOREREG. gpr: DataReg at 23:18; reg_addr: AddrLo at 17:0. */
#define FT_BH_ENC_STOREREG(gpr, reg_addr) FT_BH_ENC(0x67u, ((gpr) << 18) + ((reg_addr) << 0))
#define FT_BH_TTINSN_STOREREG(gpr, reg_addr) FT_BH_TTINSN(FT_BH_ENC_STOREREG(gpr, reg_addr))
#define FT_BH_PUSH_STOREREG(gpr, reg_addr) FT_BH_PUSH(FT_BH_ENC_STOREREG(gpr, reg_addr))

/* ===== Address counters (ADCs) =========================================================== */

/* SETADC.md (WH page), Bits32 SETADC. set_mask: PK<<2 | U1<<1 | U0 at 23:21; channel: Channel at
 * 20; dim: XYZW at 19:18; value: NewValue at 17:0. */
#define FT_BH_ENC_SETADC(set_mask, channel, dim, value)                                          \
    FT_BH_ENC(0x50u, ((set_mask) << 21) + ((channel) << 20) + ((dim) << 18) + ((value) << 0))
#define FT_BH_TTINSN_SETADC(set_mask, channel, dim, value)                                       \
    FT_BH_TTINSN(FT_BH_ENC_SETADC(set_mask, channel, dim, value))
#define FT_BH_PUSH_SETADC(set_mask, channel, dim, value)                                         \
    FT_BH_PUSH(FT_BH_ENC_SETADC(set_mask, channel, dim, value))

/* SETADCXY.md / SETADCZW.md (WH pages), Bits32 SETADCXY / SETADCZW. set_mask at 23:21; ch1 second
 * coordinate (Y1Val / W1Val) at 17:15; ch1 first (X1Val / Z1Val) at 14:12; ch0 second (Y0Val /
 * W0Val) at 11:9; ch0 first (X0Val / Z0Val) at 8:6; bit_mask (Y1 X1 Y0 X0 / W1 Z1 W0 Z0) at 3:0.
 * ThreadOverride (19:18) has no operand. */
#define FT_BH_SETADC2D_FIELDS(set_mask, ch1_b, ch1_a, ch0_b, ch0_a, bit_mask)                   \
    (((set_mask) << 21) + ((ch1_b) << 15) + ((ch1_a) << 12) + ((ch0_b) << 9) + ((ch0_a) << 6)  \
     + ((bit_mask) << 0))

#define FT_BH_ENC_SETADCXY(set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask)                       \
    FT_BH_ENC(0x51u, FT_BH_SETADC2D_FIELDS(set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask))
#define FT_BH_TTINSN_SETADCXY(set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask)                    \
    FT_BH_TTINSN(FT_BH_ENC_SETADCXY(set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask))
#define FT_BH_PUSH_SETADCXY(set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask)                      \
    FT_BH_PUSH(FT_BH_ENC_SETADCXY(set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask))

#define FT_BH_ENC_SETADCZW(set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask)                       \
    FT_BH_ENC(0x54u, FT_BH_SETADC2D_FIELDS(set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask))
#define FT_BH_TTINSN_SETADCZW(set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask)                    \
    FT_BH_TTINSN(FT_BH_ENC_SETADCZW(set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask))
#define FT_BH_PUSH_SETADCZW(set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask)                      \
    FT_BH_PUSH(FT_BH_ENC_SETADCZW(set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask))

/* SETADCXX.md (WH page), Bits32 SETADCXX. set_mask at 23:21; x_end: X1Val at 19:10; x_start:
 * X0Val at 9:0. */
#define FT_BH_ENC_SETADCXX(set_mask, x_end, x_start)                                             \
    FT_BH_ENC(0x5Eu, ((set_mask) << 21) + ((x_end) << 10) + ((x_start) << 0))
#define FT_BH_TTINSN_SETADCXX(set_mask, x_end, x_start)                                          \
    FT_BH_TTINSN(FT_BH_ENC_SETADCXX(set_mask, x_end, x_start))
#define FT_BH_PUSH_SETADCXX(set_mask, x_end, x_start)                                            \
    FT_BH_PUSH(FT_BH_ENC_SETADCXX(set_mask, x_end, x_start))

/* ===== Unpackers and packers ============================================================= */

/* UNPACR_Regular.md (BH page defers to WH, conditionalized inline), Bits32 UNPACR_Regular (no
 * _BH entry). which: WhichUnpacker at 23; addr_mod: Ch1YInc/Ch1ZInc/Ch0YInc/Ch0ZInc at 22:15;
 * ctx_count_inc: bit 13 (the IncrementContextCounter form sets it); ctx_id: ContextNumber at
 * 12:10; adc_ctx_id: ContextADC at 9:8; ovrd_thread_id: MultiContextMode at 7; set_dvalid:
 * FlipSrc at 6; zero_write2: AllDatumsAreZero at 4; auto_inc_ctx_id: UseContextCounter at 3;
 * row_search: RowSearch at 2; cache_flush: bit 1 (the FlushCache form sets it).
 * rarefy_b at 5 and last at 0 are inferred: those two bits are the only ones left undrawn, and
 * the documented operand order puts the two corresponding constant operands at those slots. */
#define FT_BH_ENC_UNPACR(which, addr_mod, ctx_count_inc, ctx_id, adc_ctx_id, ovrd_thread_id,    \
                         set_dvalid, rarefy_b, zero_write2, auto_inc_ctx_id, row_search,        \
                         cache_flush, last)                                                     \
    FT_BH_ENC(0x42u, ((which) << 23) + ((addr_mod) << 15) + ((ctx_count_inc) << 13)            \
                         + ((ctx_id) << 10) + ((adc_ctx_id) << 8) + ((ovrd_thread_id) << 7)    \
                         + ((set_dvalid) << 6) + ((rarefy_b) << 5) + ((zero_write2) << 4)      \
                         + ((auto_inc_ctx_id) << 3) + ((row_search) << 2)                      \
                         + ((cache_flush) << 1) + ((last) << 0))
#define FT_BH_TTINSN_UNPACR(which, addr_mod, ctx_count_inc, ctx_id, adc_ctx_id, ovrd_thread_id, \
                            set_dvalid, rarefy_b, zero_write2, auto_inc_ctx_id, row_search,     \
                            cache_flush, last)                                                  \
    FT_BH_TTINSN(FT_BH_ENC_UNPACR(which, addr_mod, ctx_count_inc, ctx_id, adc_ctx_id,          \
                                  ovrd_thread_id, set_dvalid, rarefy_b, zero_write2,           \
                                  auto_inc_ctx_id, row_search, cache_flush, last))
#define FT_BH_PUSH_UNPACR(which, addr_mod, ctx_count_inc, ctx_id, adc_ctx_id, ovrd_thread_id,   \
                          set_dvalid, rarefy_b, zero_write2, auto_inc_ctx_id, row_search,       \
                          cache_flush, last)                                                    \
    FT_BH_PUSH(FT_BH_ENC_UNPACR(which, addr_mod, ctx_count_inc, ctx_id, adc_ctx_id,            \
                                ovrd_thread_id, set_dvalid, rarefy_b, zero_write2,             \
                                auto_inc_ctx_id, row_search, cache_flush, last))

/* UNPACR_NOP*.md (WH pages), Bits32 UNPACR_NOP_* (no _BH entries). which: WhichUnpacker at 23;
 * stream_id: WhichStream at 21:16; msg_clr_cnt: ClearCount at 14:4; stall_clr_ctrl:
 * WaitLikeUnpacr at 4; bank_clr_ctrl: BothBanks at 3; src_clr_val_ctrl: NegativeInfSrcA at 2
 * (UNPACR_NOP_ZEROSRC.md notes this bit is part of a wider clear-value field on BH; width not
 * documented); unpack_pop: the sub-op select at 2:0 (0x2 = plain NOP).
 * set_dvalid at 8: Blackhole layout, measured by the instruction diff (a ZEROSRC-form word with
 * set_dvalid = 1 reads 0x101 in its low 12 bits). clr_to1_fmt_ctrl at 5 is inferred (an undrawn
 * bit above the ZEROSRC controls) and not yet exercised by a non-zero value. */
#define FT_BH_ENC_UNPACR_NOP(which, stream_id, msg_clr_cnt, set_dvalid, clr_to1_fmt_ctrl,       \
                             stall_clr_ctrl, bank_clr_ctrl, src_clr_val_ctrl, unpack_pop)       \
    FT_BH_ENC(0x43u, ((which) << 23) + ((stream_id) << 16) + ((set_dvalid) << 8)               \
                         + ((clr_to1_fmt_ctrl) << 5) + ((msg_clr_cnt) << 4)                    \
                         + ((stall_clr_ctrl) << 4) + ((bank_clr_ctrl) << 3)                    \
                         + ((src_clr_val_ctrl) << 2) + ((unpack_pop) << 0))
#define FT_BH_TTINSN_UNPACR_NOP(which, stream_id, msg_clr_cnt, set_dvalid, clr_to1_fmt_ctrl,    \
                                stall_clr_ctrl, bank_clr_ctrl, src_clr_val_ctrl, unpack_pop)    \
    FT_BH_TTINSN(FT_BH_ENC_UNPACR_NOP(which, stream_id, msg_clr_cnt, set_dvalid,               \
                                      clr_to1_fmt_ctrl, stall_clr_ctrl, bank_clr_ctrl,         \
                                      src_clr_val_ctrl, unpack_pop))
#define FT_BH_PUSH_UNPACR_NOP(which, stream_id, msg_clr_cnt, set_dvalid, clr_to1_fmt_ctrl,      \
                              stall_clr_ctrl, bank_clr_ctrl, src_clr_val_ctrl, unpack_pop)      \
    FT_BH_PUSH(FT_BH_ENC_UNPACR_NOP(which, stream_id, msg_clr_cnt, set_dvalid,                 \
                                    clr_to1_fmt_ctrl, stall_clr_ctrl, bank_clr_ctrl,           \
                                    src_clr_val_ctrl, unpack_pop))

/* PACR.md (BH page), Bits32 PACR_BH. cfg_ctxt: CfgContext at 22:21; row_pad_zero: RowPadZero at
 * 20:18; dst_access_mode: DstAccessMode at 17; addr_mod: AddrMod at 16:15; addr_cnt_ctxt:
 * AddrCntContext at 14:13; zero_write: ZeroWrite at 12; read_intf_sel: ReadIntfSel at 11:8;
 * ovrd_thread_id: OvrdThreadId at 7; concat: Concat at 4; ctxt_ctrl: CtxtCtrl at 3:2; flush:
 * Flush at 1; last: Last at 0. */
#define FT_BH_ENC_PACR(cfg_ctxt, row_pad_zero, dst_access_mode, addr_mod, addr_cnt_ctxt,        \
                       zero_write, read_intf_sel, ovrd_thread_id, concat, ctxt_ctrl, flush,     \
                       last)                                                                    \
    FT_BH_ENC(0x41u, ((cfg_ctxt) << 21) + ((row_pad_zero) << 18) + ((dst_access_mode) << 17)   \
                         + ((addr_mod) << 15) + ((addr_cnt_ctxt) << 13) + ((zero_write) << 12) \
                         + ((read_intf_sel) << 8) + ((ovrd_thread_id) << 7) + ((concat) << 4)  \
                         + ((ctxt_ctrl) << 2) + ((flush) << 1) + ((last) << 0))
#define FT_BH_TTINSN_PACR(cfg_ctxt, row_pad_zero, dst_access_mode, addr_mod, addr_cnt_ctxt,     \
                          zero_write, read_intf_sel, ovrd_thread_id, concat, ctxt_ctrl, flush,  \
                          last)                                                                 \
    FT_BH_TTINSN(FT_BH_ENC_PACR(cfg_ctxt, row_pad_zero, dst_access_mode, addr_mod,             \
                                addr_cnt_ctxt, zero_write, read_intf_sel, ovrd_thread_id,      \
                                concat, ctxt_ctrl, flush, last))
#define FT_BH_PUSH_PACR(cfg_ctxt, row_pad_zero, dst_access_mode, addr_mod, addr_cnt_ctxt,       \
                        zero_write, read_intf_sel, ovrd_thread_id, concat, ctxt_ctrl, flush,    \
                        last)                                                                   \
    FT_BH_PUSH(FT_BH_ENC_PACR(cfg_ctxt, row_pad_zero, dst_access_mode, addr_mod,               \
                              addr_cnt_ctxt, zero_write, read_intf_sel, ovrd_thread_id,        \
                              concat, ctxt_ctrl, flush, last))

/* ===== Vector Unit (SFPU) ================================================================ */

/* SFPLOAD.md (BH page), Bits32 SFPLOAD_BH. lreg: VD at 23:20; instr_mod: Mod0 at 19:16;
 * addr_mode: AddrMod at 15:13 (3 bits on BH; WH is 2 bits at 15:14); dest_addr: Imm10 at 9:0. */
#define FT_BH_ENC_SFPLOAD(lreg, instr_mod, addr_mode, dest_addr)                                 \
    FT_BH_ENC(0x70u, ((lreg) << 20) + ((instr_mod) << 16) + ((addr_mode) << 13)                \
                         + ((dest_addr) << 0))
#define FT_BH_TTINSN_SFPLOAD(lreg, instr_mod, addr_mode, dest_addr)                              \
    FT_BH_TTINSN(FT_BH_ENC_SFPLOAD(lreg, instr_mod, addr_mode, dest_addr))
#define FT_BH_PUSH_SFPLOAD(lreg, instr_mod, addr_mode, dest_addr)                                \
    FT_BH_PUSH(FT_BH_ENC_SFPLOAD(lreg, instr_mod, addr_mode, dest_addr))

/* SFPLOADMACRO.md (BH page), Bits32 SFPLOADMACRO_BH. lreg: MacroIndex<<2 | VDLo at 23:20;
 * instr_mod: Mod0 at 19:16; addr_mode: AddrMod at 15:13 (WH: 2 bits at 15:14); dest_addr:
 * Imm9<<1 | VDHi at 9:0. */
#define FT_BH_ENC_SFPLOADMACRO(lreg, instr_mod, addr_mode, dest_addr)                            \
    FT_BH_ENC(0x93u, ((lreg) << 20) + ((instr_mod) << 16) + ((addr_mode) << 13)                \
                         + ((dest_addr) << 0))
#define FT_BH_TTINSN_SFPLOADMACRO(lreg, instr_mod, addr_mode, dest_addr)                         \
    FT_BH_TTINSN(FT_BH_ENC_SFPLOADMACRO(lreg, instr_mod, addr_mode, dest_addr))
#define FT_BH_PUSH_SFPLOADMACRO(lreg, instr_mod, addr_mode, dest_addr)                           \
    FT_BH_PUSH(FT_BH_ENC_SFPLOADMACRO(lreg, instr_mod, addr_mode, dest_addr))

/* SFPLOADI.md (BH page defers to WH, identical), Bits32 SFPLOADI. lreg: VD at 23:20; instr_mod:
 * Mod0 at 19:16; imm16: Imm16 at 15:0. */
#define FT_BH_ENC_SFPLOADI(lreg, instr_mod, imm16)                                               \
    FT_BH_ENC(0x71u, ((lreg) << 20) + ((instr_mod) << 16) + ((imm16) << 0))
#define FT_BH_TTINSN_SFPLOADI(lreg, instr_mod, imm16)                                            \
    FT_BH_TTINSN(FT_BH_ENC_SFPLOADI(lreg, instr_mod, imm16))
#define FT_BH_PUSH_SFPLOADI(lreg, instr_mod, imm16)                                              \
    FT_BH_PUSH(FT_BH_ENC_SFPLOADI(lreg, instr_mod, imm16))

/* SFPSHFT2.md (BH page), Bits32 SFPSHFT2 / SFPSHFT2b. imm12: VB or Imm12 at 23:12; lreg_c: VC at
 * 11:8; lreg_dest: VD at 7:4; instr_mod: Mod1 at 3:0. */
#define FT_BH_ENC_SFPSHFT2(imm12, lreg_c, lreg_dest, instr_mod)                                  \
    FT_BH_ENC(0x94u, ((imm12) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4)                    \
                         + ((instr_mod) << 0))
#define FT_BH_TTINSN_SFPSHFT2(imm12, lreg_c, lreg_dest, instr_mod)                               \
    FT_BH_TTINSN(FT_BH_ENC_SFPSHFT2(imm12, lreg_c, lreg_dest, instr_mod))
#define FT_BH_PUSH_SFPSHFT2(imm12, lreg_c, lreg_dest, instr_mod)                                 \
    FT_BH_PUSH(FT_BH_ENC_SFPSHFT2(imm12, lreg_c, lreg_dest, instr_mod))

/* SFPSWAP.md (BH page), Bits32 SFPSWAP. imm12: first operand (0 in the documented syntax),
 * placed at 23:12 like SFPSHFT2; lreg_c: VC at 11:8; lreg_dest: VD at 7:4; instr_mod: Mod1 at
 * 3:0. */
#define FT_BH_ENC_SFPSWAP(imm12, lreg_c, lreg_dest, instr_mod)                                   \
    FT_BH_ENC(0x92u, ((imm12) << 12) + ((lreg_c) << 8) + ((lreg_dest) << 4)                    \
                         + ((instr_mod) << 0))
#define FT_BH_TTINSN_SFPSWAP(imm12, lreg_c, lreg_dest, instr_mod)                                \
    FT_BH_TTINSN(FT_BH_ENC_SFPSWAP(imm12, lreg_c, lreg_dest, instr_mod))
#define FT_BH_PUSH_SFPSWAP(imm12, lreg_c, lreg_dest, instr_mod)                                  \
    FT_BH_PUSH(FT_BH_ENC_SFPSWAP(imm12, lreg_c, lreg_dest, instr_mod))

/* SFPCONFIG.md (BH page defers to WH, identical), Bits32 SFPCONFIG. imm16: Imm16 at 23:8;
 * config_dest: VD at 7:4; instr_mod: Mod1 at 3:0. */
#define FT_BH_ENC_SFPCONFIG(imm16, config_dest, instr_mod)                                       \
    FT_BH_ENC(0x91u, ((imm16) << 8) + ((config_dest) << 4) + ((instr_mod) << 0))
#define FT_BH_TTINSN_SFPCONFIG(imm16, config_dest, instr_mod)                                    \
    FT_BH_TTINSN(FT_BH_ENC_SFPCONFIG(imm16, config_dest, instr_mod))
#define FT_BH_PUSH_SFPCONFIG(imm16, config_dest, instr_mod)                                      \
    FT_BH_PUSH(FT_BH_ENC_SFPCONFIG(imm16, config_dest, instr_mod))

/* ===== Field-value constants ============================================================= */

/* SETRWC.md: clear_dvalid (FlipSrcB<<1 | FlipSrcA), set_mask (Fidelity Dst SrcB SrcA) and cr
 * (DstCtoCr DstCr SrcBCr SrcACr) bit values. */
#define FT_BH_SETRWC_CLR_NONE 0x0u
#define FT_BH_SETRWC_CLR_A 0x1u
#define FT_BH_SETRWC_CLR_B 0x2u
#define FT_BH_SETRWC_CLR_AB 0x3u
#define FT_BH_SETRWC_SET_A 0x1u
#define FT_BH_SETRWC_SET_B 0x2u
#define FT_BH_SETRWC_SET_AB 0x3u
#define FT_BH_SETRWC_SET_D 0x4u
#define FT_BH_SETRWC_SET_ABD_F 0xFu
#define FT_BH_SETRWC_CR_AB 0x3u
#define FT_BH_SETRWC_CR_D 0x4u

/* STALLWAIT.md (BH page) condition bits C0..C12; SEMWAIT-shared names. */
#define FT_BH_STALLWAIT_COND_NONE 0x0u
#define FT_BH_STALLWAIT_COND_THCON 0x1u
#define FT_BH_STALLWAIT_COND_UNPACK0 0x2u
#define FT_BH_STALLWAIT_COND_UNPACK1 0x4u
#define FT_BH_STALLWAIT_COND_UNPACK 0x6u
#define FT_BH_STALLWAIT_COND_PACK 0x8u
#define FT_BH_STALLWAIT_COND_MATH 0x10u
#define FT_BH_STALLWAIT_COND_SRCA_VALID 0x80u
#define FT_BH_STALLWAIT_COND_SRCB_VALID 0x100u
#define FT_BH_STALLWAIT_COND_TRISC_CFG 0x400u
#define FT_BH_STALLWAIT_COND_SFPU1 0x800u
#define FT_BH_STALLWAIT_COND_SFPU 0x800u

/* STALLWAIT.md (BH page) block bits B0..B8. */
#define FT_BH_STALLWAIT_BLOCK_TDMA 0x1u
#define FT_BH_STALLWAIT_BLOCK_SYNC 0x2u
#define FT_BH_STALLWAIT_BLOCK_UNPACK 0x8u
#define FT_BH_STALLWAIT_BLOCK_THCON 0x20u
#define FT_BH_STALLWAIT_BLOCK_MATH 0x40u
#define FT_BH_STALLWAIT_BLOCK_CFG 0x80u
#define FT_BH_STALLWAIT_BLOCK_SFPU 0x100u

/* SEMWAIT.md condition bits C0 (Value == 0) and C1 (Value >= Max); semaphore 1 select bit. */
#define FT_BH_SEMWAIT_COND_ZERO 0x1u
#define FT_BH_SEMWAIT_COND_MAX 0x2u
#define FT_BH_SEMAPHORE_1 0x2u

/* ELWADD.md srcb_bcast values (BroadcastSrcBRow<<1 | BroadcastSrcBCol0). */
#define FT_BH_ELWADD_SRCB_NO_BCAST 0x0u
#define FT_BH_ELWADD_SRCB_BCAST_COL 0x1u
#define FT_BH_ELWADD_SRCB_BCAST_ROW 0x2u
#define FT_BH_ELWADD_SRCB_BCAST_ALL 0x3u

/* ZEROACC.md Mode values. */
#define FT_BH_ZEROACC_CLR_SPECIFIC 0x0u
#define FT_BH_ZEROACC_CLR_16 0x1u
#define FT_BH_ZEROACC_CLR_HALF 0x2u
#define FT_BH_ZEROACC_CLR_ALL 0x3u

/* SETADC.md set_mask (PK U1 U0), XYZW index for W, channel 0. */
#define FT_BH_SETADC_UNP_A 0x1u
#define FT_BH_SETADC_UNP_B 0x2u
#define FT_BH_SETADC_UNP_AB 0x3u
#define FT_BH_SETADC_PAC 0x4u
#define FT_BH_SETADC_SET_W 0x3u
#define FT_BH_SETADC_CH_0 0x0u

/* UNPACR_Regular.md: rarefy-B off; WhichUnpacker 0 targets SrcA, 1 targets SrcB. */
#define FT_BH_UNPACR_RAREFYB_DISABLE 0x0u
#define FT_BH_UNPACR_SRC_A 0x0u
#define FT_BH_UNPACR_SRC_B 0x1u

/* PACR.md (BH page) operand values. */
#define FT_BH_PACR_ZERO_OUTPUT_DISABLED 0x0u
#define FT_BH_PACR_DST_ACCESS_NORMAL_MODE 0x0u
#define FT_BH_PACR_NO_ROW_PAD_ZERO 0x0u
#define FT_BH_PACR_CFG_CTXT_0 0x0u
#define FT_BH_PACR_ADDR_CNT_CTXT_0 0x0u
#define FT_BH_PACR_ALL_INTF_ACTIVE 0x0u
#define FT_BH_PACR_SINGLE_INTF_ACTIVE 0x1u
#define FT_BH_PACR_TWO_INTFS_ACTIVE 0x3u
#define FT_BH_PACR_NO_CTXT_CTRL 0x0u

/* WRCFG.md Is128Bit. */
#define FT_BH_WRCFG_32B 0x0u
#define FT_BH_WRCFG_128B 0x1u

/* Hardware data-format codes (Unpackers/FormatConversion.md type-encoding table for the 4-bit
 * codes; UInt32/Fp8_e4m3/UInt8 use the fifth bit, kept as given by the spec). */
#define FT_BH_DATA_FORMAT_FLOAT32 0u
#define FT_BH_DATA_FORMAT_FLOAT16 1u
#define FT_BH_DATA_FORMAT_BFP8 2u
#define FT_BH_DATA_FORMAT_TF32 4u
#define FT_BH_DATA_FORMAT_FLOAT16_B 5u
#define FT_BH_DATA_FORMAT_BFP8_B 6u
#define FT_BH_DATA_FORMAT_INT32 8u
#define FT_BH_DATA_FORMAT_UINT16 9u
#define FT_BH_DATA_FORMAT_LF8 10u
#define FT_BH_DATA_FORMAT_INT8 14u
#define FT_BH_DATA_FORMAT_UINT32 24u
#define FT_BH_DATA_FORMAT_FP8_E4M3 26u
#define FT_BH_DATA_FORMAT_UINT8 30u
#define FT_BH_DATA_FORMAT_CONFIG_MASK 0xFu
#define FT_BH_DATA_FORMAT_CONFIG_BITS(f) ((f) & 0xF)

/* Address-modifier slot indices (RWCs.md). */
#define FT_BH_ADDR_MOD_0 0u
#define FT_BH_ADDR_MOD_1 1u
#define FT_BH_ADDR_MOD_2 2u
#define FT_BH_ADDR_MOD_3 3u
#define FT_BH_ADDR_MOD_4 4u
#define FT_BH_ADDR_MOD_5 5u
#define FT_BH_ADDR_MOD_6 6u
#define FT_BH_ADDR_MOD_7 7u

#endif /* FT_BH_ISA_H */
