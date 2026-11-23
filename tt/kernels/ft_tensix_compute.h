/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute.h
 * @brief Base header for Foundry's C compute kernels (TRISC0 unpack, TRISC1 math, TRISC2
 *   pack): thread gating, Tensix register windows, firmware-shared globals, sync and
 *   semaphore helpers, masked config writes, the MOP template and ADDR_MOD slot helpers.
 *
 * One kernel source is compiled three times; exactly one of FT_TRISC_UNPACK / FT_TRISC_MATH /
 * FT_TRISC_PACK is defined per build, and the FT_TENSIX_UNPACK/MATH/PACK wrappers keep each
 * statement on its own thread. The register map (tt/fw/ft_bh_regs.h) and the instruction
 * encoders (tt/fw/ft_bh_isa.h) are made visible to every header that includes this one.
 *
 * Addresses: BlackholeA0/TensixTile/BabyRISCV/README.md (TRISC memory map: GPR file, PC
 * buffer, config space, MOP config). Semaphore/GPR/mutex assignments are Foundry convention.
 *
 * @status Clean-room rewrite, plan Phase 4 step 4.5a.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_H
#define FT_TENSIX_COMPUTE_H

#include <stdint.h>

#include "../fw/ft_bh_regs.h"
#include "../fw/ft_bh_isa.h"

/* ===== Thread gating ===================================================================== */

#ifdef FT_TRISC_UNPACK
#define FT_TENSIX_UNPACK(...) __VA_ARGS__
#else
#define FT_TENSIX_UNPACK(...)
#endif

#ifdef FT_TRISC_MATH
#define FT_TENSIX_MATH(...) __VA_ARGS__
#else
#define FT_TENSIX_MATH(...)
#endif

#ifdef FT_TRISC_PACK
#define FT_TENSIX_PACK(...) __VA_ARGS__
#else
#define FT_TENSIX_PACK(...)
#endif

/* ===== Register windows (TRISC memory map) =============================================== */

/* GPR file: word i is GPR i. */
#define FT_TENSIX_REGFILE_BASE 0xFFE00000u
/* PC buffer: word 1 = sync, word 2 = MOP sync, words 8.. = semaphores. */
#define FT_TENSIX_PC_BUF_BASE 0xFFE80000u
/* Backend config space, state 0. */
#define FT_TENSIX_CFG_BASE 0xFFEF0000u
/* MOP expander config registers. */
#define FT_TENSIX_MOP_CFG_BASE 0xFFB80000u

#define FT_TENSIX_REGFILE ((volatile uint32_t FT_BH_REG_PTR *)FT_TENSIX_REGFILE_BASE)
#define FT_TENSIX_PC_BUF ((volatile uint32_t FT_BH_REG_PTR *)FT_TENSIX_PC_BUF_BASE)
#define FT_TENSIX_MOP_CFG ((volatile uint32_t *)FT_TENSIX_MOP_CFG_BASE)

/* ===== Geometry ========================================================================== */

#define FT_TENSIX_FACE_R_DIM 16u
#define FT_TENSIX_FACE_C_DIM 16u
#define FT_TENSIX_TILE_NUM_FACES 4u
#define FT_TENSIX_MAX_FPU_ROWS 8u
#define FT_TENSIX_MAX_FPU_ROWS_LOG2 3u
#define FT_TENSIX_DEST_REGISTER_FULL_SIZE (64u * 16u)
#define FT_TENSIX_DEST_REGISTER_HALF_SIZE (FT_TENSIX_DEST_REGISTER_FULL_SIZE / 2u)

/* ===== Semaphores, mutexes, GPRs (Foundry convention) ==================================== */

#define FT_TENSIX_SEM_MATH_PACK 1u
#define FT_TENSIX_SEM_UNPACK_SYNC 5u
#define FT_TENSIX_T6_SEM(index) (1u << (index))
#define FT_TENSIX_PC_BUF_SEMAPHORE_BASE 8u
#define FT_TENSIX_MUTEX_REG_RMW 0u

#define FT_TENSIX_GPR_ZERO 0u
#define FT_TENSIX_GPR_PACK_DEST_OFFSET_LO 4u
#define FT_TENSIX_GPR_PACK_DEST_OFFSET_HI 8u
#define FT_TENSIX_GPR_PACK_OUTPUT_ADDR 12u
#define FT_TENSIX_GPR_PACK_TILE_HEADER 16u
#define FT_TENSIX_GPR_PACK_NUM_MSGS_RECEIVED 24u
#define FT_TENSIX_GPR_PACK_TMP0 28u
#define FT_TENSIX_GPR_PACK_TMP1 29u
#define FT_TENSIX_GPR_UNPACK_TILE_SIZE_A 36u
#define FT_TENSIX_GPR_UNPACK_TILE_SIZE_B 37u
#define FT_TENSIX_GPR_UNPACK_FACE_DIM_16x16 40u
#define FT_TENSIX_GPR_UNPACK_FACE_DIM_8x16 41u
#define FT_TENSIX_GPR_UNPACK_FACE_DIM_4x16 42u
#define FT_TENSIX_GPR_UNPACK_FACE_DIM_2x16 43u
#define FT_TENSIX_GPR_UNPACK_FACE_DIM_1x16 44u
#define FT_TENSIX_GPR_PACK_EXP0_SEC_SIZE_BFP 52u

/* SETDMAREG addresses 16-bit halves of the GPRs: half 2*g is the low half of GPR g. */
#define FT_TENSIX_LO_16(gpr) ((gpr) * 2u)
#define FT_TENSIX_HI_16(gpr) ((gpr) * 2u + 1u)
#define FT_TENSIX_LOWER_HALFWORD(x) ((x) & 0xFFFFu)
#define FT_TENSIX_UPPER_HALFWORD(x) ((x) >> 16)

/* ===== Firmware-owned globals ============================================================ */
/* Defined by the TRISC firmware (fw/ft_trisc_fw.c); a kernel binds to the same objects through
 * --just-symbols, so the link names are part of the contract. They were chosen by
 * tests/probe_link_name_sort.sh: GCC's register allocation follows global symbol names here,
 * and these leave every image LOAD-IDENTICAL (8 other pairs tried moved 1-5 images). */

extern uint32_t ft_tensix_cfg_state_id __asm__("tensix_cfg_state_id");
extern uint32_t ft_tensix_dest_offset_id __asm__("tensix_dest_offset_id");
extern uint32_t unp_cfg_context;
extern uint32_t pack_sync_tile_dst_ptr;
extern uint32_t math_sync_tile_dst_index;

/* ===== Sync helpers ====================================================================== */

/** Store `value` to `ptr`, read it back into the same register and consume it, so the store
 *  has completed before the next instruction. */
static FT_BH_INLINE void ft_tensix_store_blocking(volatile uint32_t FT_BH_REG_PTR *ptr,
                                                  uint32_t value)
{
    __asm__ __volatile__("sw %0, 0(%1)\n\t"
                         "lw %0, 0(%1)\n\t"
                         "and x0, x0, %0"
                         : "+r"(value)
                         : "r"(ptr)
                         : "memory");
}

/** Wait for the coprocessor to accept everything this thread has issued. */
static FT_BH_INLINE void ft_tensix_sync(void)
{
    ft_tensix_store_blocking(&FT_TENSIX_PC_BUF[1], 0);
}

/** Wait for the MOP expander to go idle. */
static FT_BH_INLINE void ft_tensix_mop_sync(void)
{
    ft_tensix_store_blocking(&FT_TENSIX_PC_BUF[2], 0);
}

/* ===== RISC-side semaphores ============================================================== */

static FT_BH_INLINE uint8_t ft_tensix_semaphore_read(uint8_t index)
{
    return (uint8_t)FT_TENSIX_PC_BUF[FT_TENSIX_PC_BUF_SEMAPHORE_BASE + index];
}

/** A store of 0 posts (increments) the semaphore. */
static FT_BH_INLINE void ft_tensix_semaphore_post(uint8_t index)
{
    FT_TENSIX_PC_BUF[FT_TENSIX_PC_BUF_SEMAPHORE_BASE + index] = 0;
}

/* ===== Config space ====================================================================== */

/** Base of the config state this thread currently uses. */
static FT_BH_INLINE volatile uint32_t FT_BH_REG_PTR *ft_tensix_cfg_pointer(void)
{
    if (ft_tensix_cfg_state_id == 0) {
        return (volatile uint32_t FT_BH_REG_PTR *)FT_TENSIX_CFG_BASE;
    }
    return (volatile uint32_t FT_BH_REG_PTR *)(FT_TENSIX_CFG_BASE + FT_BH_CFG_STATE_SIZE * 16u);
}

/* Masked read-modify-write of one config word, one byte lane per RMWCIBn (RMWCIB.md); lanes
 * whose mask byte is zero issue nothing. Always the instruction-buffer form. */
#define FT_TENSIX_CFG_RMW(cfg_addr32, shamt, mask, val)                                         \
    do {                                                                                        \
        uint32_t ft_cfg_rmw_data = (uint32_t)(val) << (shamt);                                  \
        if (((mask) & 0xffu) != 0u) {                                                           \
            FT_BH_PUSH_RMWCIB0(((mask) & 0xffu), (ft_cfg_rmw_data & 0xffu), (cfg_addr32));      \
        }                                                                                       \
        ft_cfg_rmw_data >>= 8;                                                                  \
        if ((((mask) >> 8) & 0xffu) != 0u) {                                                    \
            FT_BH_PUSH_RMWCIB1((((mask) >> 8) & 0xffu), (ft_cfg_rmw_data & 0xffu),             \
                               (cfg_addr32));                                                   \
        }                                                                                       \
        ft_cfg_rmw_data >>= 8;                                                                  \
        if ((((mask) >> 16) & 0xffu) != 0u) {                                                   \
            FT_BH_PUSH_RMWCIB2((((mask) >> 16) & 0xffu), (ft_cfg_rmw_data & 0xffu),            \
                               (cfg_addr32));                                                   \
        }                                                                                       \
        ft_cfg_rmw_data >>= 8;                                                                  \
        if ((((mask) >> 24) & 0xffu) != 0u) {                                                   \
            FT_BH_PUSH_RMWCIB3((((mask) >> 24) & 0xffu), (ft_cfg_rmw_data & 0xffu),            \
                               (cfg_addr32));                                                   \
        }                                                                                       \
    } while (0)

/* RMW of one named field F of ft_bh_regs.h (FT_BH_CFG_F_WORD/_SHIFT/_MASK). */
#define FT_TENSIX_CFG_RMW_FIELD(field, val)                                                     \
    FT_TENSIX_CFG_RMW(FT_BH_CFG_##field##_WORD, FT_BH_CFG_##field##_SHIFT,                      \
                      FT_BH_CFG_##field##_MASK, (val))

/** Make earlier GPR-file writes land by reading GPR `index` back into a volatile local. */
static FT_BH_INLINE void ft_tensix_sync_regfile_write(uint32_t index)
{
    volatile uint32_t sink = 0;
    volatile uint32_t *sink_ptr = &sink;
    *sink_ptr = FT_TENSIX_REGFILE[index];
}

/* ===== MOP template (MOPExpander.md, double-loop template) =============================== */

typedef struct {
    uint32_t outer_loop_len;
    uint32_t inner_loop_len;
    uint32_t start_op0;
    uint32_t end_op0;
    uint32_t end_op1;
    uint32_t loop_op0;
    uint32_t loop_op1;
    uint32_t loop0_last_instr;
    uint32_t loop1_last_instr;
} ft_tensix_mop_template;

/** Template with one loop instruction. */
static FT_BH_INLINE ft_tensix_mop_template ft_tensix_mop_template_1(uint32_t outer_loop_len,
                                                                     uint32_t inner_loop_len,
                                                                     uint32_t loop_op)
{
    ft_tensix_mop_template t;
    t.outer_loop_len = outer_loop_len;
    t.inner_loop_len = inner_loop_len;
    t.start_op0 = FT_BH_ENC_NOP;
    t.end_op0 = FT_BH_ENC_NOP;
    t.end_op1 = FT_BH_ENC_NOP;
    t.loop_op0 = loop_op;
    t.loop_op1 = FT_BH_ENC_NOP;
    t.loop0_last_instr = loop_op;
    t.loop1_last_instr = loop_op;
    return t;
}

/** Template with two loop instructions; both last-iteration slots take the second. */
static FT_BH_INLINE ft_tensix_mop_template ft_tensix_mop_template_2(uint32_t outer_loop_len,
                                                                     uint32_t inner_loop_len,
                                                                     uint32_t loop_op0,
                                                                     uint32_t loop_op1)
{
    ft_tensix_mop_template t;
    t.outer_loop_len = outer_loop_len;
    t.inner_loop_len = inner_loop_len;
    t.start_op0 = FT_BH_ENC_NOP;
    t.end_op0 = FT_BH_ENC_NOP;
    t.end_op1 = FT_BH_ENC_NOP;
    t.loop_op0 = loop_op0;
    t.loop_op1 = loop_op1;
    t.loop0_last_instr = loop_op1;
    t.loop1_last_instr = loop_op1;
    return t;
}

/** Load a template into the MOP expander (register order is the hardware's). */
static FT_BH_INLINE void ft_tensix_mop_template_program(const ft_tensix_mop_template *t)
{
    volatile uint32_t *mop = FT_TENSIX_MOP_CFG;
    ft_tensix_mop_sync();
    mop[0] = t->outer_loop_len;
    mop[1] = t->inner_loop_len;
    mop[2] = t->start_op0;
    mop[3] = t->end_op0;
    mop[4] = t->end_op1;
    mop[5] = t->loop_op0;
    mop[6] = t->loop_op1;
    mop[7] = t->loop0_last_instr;
    mop[8] = t->loop1_last_instr;
}

/** Run the programmed double-loop template once. */
static FT_BH_INLINE void ft_tensix_mop_template_run(void)
{
    FT_BH_TTINSN_MOP(1, 0, 0);
}

/* ===== ADDR_MOD slots (RWCs.md) ========================================================== */

#define FT_TENSIX_ADDR_MOD_SRC_VAL(incr, clr, cr)                                               \
    (((incr) & 0x3Fu) | (((cr) & 0x1u) << 6) | (((clr) & 0x1u) << 7))
#define FT_TENSIX_ADDR_MOD_DEST_VAL(incr, clr, cr, c_to_cr)                                     \
    (((uint32_t)(incr) & 0x3FFu) | (((cr) & 0x1u) << 10) | (((clr) & 0x1u) << 11)               \
     | (((c_to_cr) & 0x1u) << 12))
#define FT_TENSIX_ADDR_MOD_FIDELITY_VAL(incr, clr) (((incr) & 0x3u) | (((clr) & 0x1u) << 2))
#define FT_TENSIX_ADDR_MOD_BIAS_VAL(incr, clr) (((incr) & 0xFu) | (((clr) & 0x1u) << 4))

/* Config words of slot s: the per-slot words are consecutive (asserted below). */
#define FT_TENSIX_ADDR_MOD_SRC_WORD(slot) (FT_BH_CFG_ADDRMOD0_SRCA_INCREMENT_WORD + (slot))
#define FT_TENSIX_ADDR_MOD_DST_WORD(slot) (FT_BH_CFG_ADDRMOD0_DEST_INCREMENT_WORD + (slot))
#define FT_TENSIX_ADDR_MOD_BIAS_WORD(slot) (FT_BH_CFG_ADDRMOD0_BIAS_INCREMENT_WORD + (slot))
#define FT_TENSIX_ADDR_MOD_PACK_WORD(slot)                                                      \
    (FT_BH_CFG_PACK_ADDRMOD0_Y_SOURCE_INCREMENT_WORD + (slot))

_Static_assert(FT_BH_CFG_ADDRMOD3_SRCA_INCREMENT_WORD == FT_BH_CFG_ADDRMOD0_SRCA_INCREMENT_WORD + 3,
               "SrcA ADDR_MOD words not consecutive");
_Static_assert(FT_BH_CFG_ADDRMOD3_DEST_INCREMENT_WORD == FT_BH_CFG_ADDRMOD0_DEST_INCREMENT_WORD + 3,
               "Dest ADDR_MOD words not consecutive");
_Static_assert(FT_BH_CFG_ADDRMOD3_BIAS_INCREMENT_WORD == FT_BH_CFG_ADDRMOD0_BIAS_INCREMENT_WORD + 3,
               "Bias ADDR_MOD words not consecutive");
_Static_assert(FT_BH_CFG_PACK_ADDRMOD3_Y_SOURCE_INCREMENT_WORD
                   == FT_BH_CFG_PACK_ADDRMOD0_Y_SOURCE_INCREMENT_WORD + 3,
               "Pack ADDR_MOD words not consecutive");
_Static_assert(FT_BH_CFG_ADDRMOD7_SRCA_INCREMENT_WORD == FT_BH_CFG_ADDRMOD0_SRCA_INCREMENT_WORD + 7,
               "SrcA ADDR_MOD words not consecutive");
_Static_assert(FT_BH_CFG_ADDRMOD7_DEST_INCREMENT_WORD == FT_BH_CFG_ADDRMOD0_DEST_INCREMENT_WORD + 7,
               "Dest ADDR_MOD words not consecutive");
_Static_assert(FT_BH_CFG_ADDRMOD7_BIAS_INCREMENT_WORD == FT_BH_CFG_ADDRMOD0_BIAS_INCREMENT_WORD + 7,
               "Bias ADDR_MOD words not consecutive");

/* Program math ADDR_MOD slot `slot`: three immediate SETC16s, so every argument must be a
 * compile-time constant after inlining. */
#define FT_TENSIX_ADDR_MOD_SET(slot, srca_val, srcb_val, dest_val, fidelity_val, bias_val)      \
    do {                                                                                        \
        FT_BH_TTINSN_SETC16(FT_TENSIX_ADDR_MOD_SRC_WORD(slot), (srca_val) | ((srcb_val) << 8)); \
        FT_BH_TTINSN_SETC16(FT_TENSIX_ADDR_MOD_DST_WORD(slot),                                  \
                            (dest_val) | ((fidelity_val) << 13));                               \
        FT_BH_TTINSN_SETC16(FT_TENSIX_ADDR_MOD_BIAS_WORD(slot), (bias_val));                    \
    } while (0)

/* Pack ADDR_MOD slot values (slots 0..3). */
#define FT_TENSIX_ADDR_MOD_PACK_VALS(incr, clr, cr)                                             \
    (((incr) & 0xFu) | (((cr) & 0x1u) << 4) | (((clr) & 0x1u) << 5))
#define FT_TENSIX_ADDR_MOD_PACK_REDUCED(incr, clr) (((incr) & 0x1u) | (((clr) & 0x1u) << 1))
#define FT_TENSIX_ADDR_MOD_PACK_VAL(y_src, y_dst, z_src, z_dst)                                 \
    ((y_src) | ((y_dst) << 6) | ((z_src) << 12) | ((z_dst) << 14))
#define FT_TENSIX_ADDR_MOD_PACK_SET(slot, value)                                                \
    FT_BH_TTINSN_SETC16(FT_TENSIX_ADDR_MOD_PACK_WORD(slot), (value))

#endif /* FT_TENSIX_COMPUTE_H */
