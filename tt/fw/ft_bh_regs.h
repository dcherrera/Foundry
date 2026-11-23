/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_bh_regs.h
 * @brief Blackhole register map for code running on a Tensix tile's RISC-V cores: compiler
 *   attributes, NoC initiator registers and NoC address packing, per-processor command-buffer
 *   assignment, overlay stream registers, the wall clock, backend configuration field
 *   positions, and the primitive register accesses built on them.
 *
 * Sources of the numbers: tt-isa-documentation BlackholeA0 pages (WormholeB0 where Blackhole
 * has no page; Diagrams/Src/Bits32.lua for bit layouts), the Blackhole cfg register table for the
 * backend configuration fields ("cfg register table"), and values Foundry chooses itself
 * ("Foundry-defined"). Each group below cites its source.
 *
 * @status Clean-room register map for Blackhole Tensix RISC-V code.
 * @issues None
 * @todo None
 */
#ifndef FT_BH_REGS_H
#define FT_BH_REGS_H

#include <stdint.h>

/* ---------------------------------------------------------------------------------------------
 * Compiler attributes (SFPI GCC). The attribute arguments change emitted code: keep them exact.
 * ------------------------------------------------------------------------------------------- */

#define FT_BH_L1_PTR  __attribute__((rvtt_l1_ptr))
#define FT_BH_REG_PTR __attribute__((rvtt_reg_ptr))
#define FT_BH_INLINE  inline __attribute__((always_inline))

/* ---------------------------------------------------------------------------------------------
 * NoC initiator registers (BlackholeA0/NoC/MemoryMap.md, "NIU Request Initiators").
 * ------------------------------------------------------------------------------------------- */

/* NIU_BASE of NoC 0 in a Tensix tile; initiator (command buffer) 0 starts here. */
#define FT_BH_NOC_REG_BASE 0xFFB20000u

/* Initiator i sits at NIU_BASE + i * 0x800 (address bit 11); NoC 1's NIU_BASE is NoC 0's plus
 * 0x10000 (address bit 16). */
#define FT_BH_NOC_CMD_BUF_SELECT_BIT 11u
#define FT_BH_NOC_INSTANCE_SELECT_BIT 16u

/* Register offsets within one initiator. */
#define FT_BH_NOC_TARGET_ADDR_LOW  0x00u
#define FT_BH_NOC_TARGET_ADDR_MID  0x04u
#define FT_BH_NOC_TARGET_ADDR_HIGH 0x08u
#define FT_BH_NOC_RETURN_ADDR_LOW  0x0Cu
#define FT_BH_NOC_RETURN_ADDR_MID  0x10u
#define FT_BH_NOC_RETURN_ADDR_HIGH 0x14u
#define FT_BH_NOC_PACKET_CONTROL   0x1Cu
#define FT_BH_NOC_ATOMIC_LEN_BE    0x20u
#define FT_BH_NOC_ATOMIC_DATA      0x28u
#define FT_BH_NOC_COMMAND_CONTROL  0x40u

/* The X/Y coordinates live in the HIGH address registers (MemoryMap.md, NOC_TARG_ADDR_HI and
 * NOC_RET_ADDR_HI). */
#define FT_BH_NOC_TARGET_COORDINATE FT_BH_NOC_TARGET_ADDR_HIGH
#define FT_BH_NOC_RETURN_COORDINATE FT_BH_NOC_RETURN_ADDR_HIGH

/* Absolute register addresses of initiator 0 on NoC 0 (FT_BH_NOC_REG_BASE + offset): the `reg`
 * argument of ft_bh_noc_cmd_write / ft_bh_noc_cmd_read. */
#define FT_BH_NOC_REG_TARGET_ADDR_LOW   (FT_BH_NOC_REG_BASE + FT_BH_NOC_TARGET_ADDR_LOW)
#define FT_BH_NOC_REG_TARGET_ADDR_MID   (FT_BH_NOC_REG_BASE + FT_BH_NOC_TARGET_ADDR_MID)
#define FT_BH_NOC_REG_TARGET_ADDR_HIGH  (FT_BH_NOC_REG_BASE + FT_BH_NOC_TARGET_ADDR_HIGH)
#define FT_BH_NOC_REG_RETURN_ADDR_LOW   (FT_BH_NOC_REG_BASE + FT_BH_NOC_RETURN_ADDR_LOW)
#define FT_BH_NOC_REG_RETURN_ADDR_MID   (FT_BH_NOC_REG_BASE + FT_BH_NOC_RETURN_ADDR_MID)
#define FT_BH_NOC_REG_RETURN_ADDR_HIGH  (FT_BH_NOC_REG_BASE + FT_BH_NOC_RETURN_ADDR_HIGH)
#define FT_BH_NOC_REG_PACKET_CONTROL    (FT_BH_NOC_REG_BASE + FT_BH_NOC_PACKET_CONTROL)
#define FT_BH_NOC_REG_ATOMIC_LEN_BE     (FT_BH_NOC_REG_BASE + FT_BH_NOC_ATOMIC_LEN_BE)
#define FT_BH_NOC_REG_ATOMIC_DATA       (FT_BH_NOC_REG_BASE + FT_BH_NOC_ATOMIC_DATA)
#define FT_BH_NOC_REG_COMMAND_CONTROL   (FT_BH_NOC_REG_BASE + FT_BH_NOC_COMMAND_CONTROL)
#define FT_BH_NOC_REG_TARGET_COORDINATE (FT_BH_NOC_REG_BASE + FT_BH_NOC_TARGET_COORDINATE)
#define FT_BH_NOC_REG_RETURN_COORDINATE (FT_BH_NOC_REG_BASE + FT_BH_NOC_RETURN_COORDINATE)

/* Status counters (BlackholeA0/NoC/Counters.md): a 32-bit array at NIU_BASE + 0x200. */
#define FT_BH_NOC_STATUS_ADDR(counter) (FT_BH_NOC_REG_BASE + 0x200u + 4u * (counter))
#define FT_BH_NOC_COUNTER_ATOMIC_RESPONSES_RECEIVED 0x0u
#define FT_BH_NOC_COUNTER_WRITE_ACKS_RECEIVED       0x1u
#define FT_BH_NOC_COUNTER_READ_RESPONSES_RECEIVED   0x2u
#define FT_BH_NOC_COUNTER_NONPOSTED_WRITES_SENT     0xAu
#define FT_BH_NOC_COUNTER_POSTED_WRITES_SENT        0xBu

/* Command control (MemoryMap.md, NOC_CMD_CTRL): writing bit 0 sends the request; the bit reads
 * back 0 once the initiator may be reprogrammed. */
#define FT_BH_NOC_COMMAND_SEND         0x1u
#define FT_BH_NOC_COMMAND_STATUS_READY 0x0u

/* Packet control bits (MemoryMap.md, NOC_CTRL). The 2-bit request type is read 0, atomic 1,
 * write 2; "copy" names the non-atomic case of bit 0 and "read" the non-write case of bit 1. */
#define FT_BH_NOC_PACKET_ATOMIC          (1u << 0u)
#define FT_BH_NOC_PACKET_COPY            (0u << 0u)
#define FT_BH_NOC_PACKET_READ            (0u << 1u)
#define FT_BH_NOC_PACKET_WRITE           (1u << 1u)
#define FT_BH_NOC_PACKET_WRITE_INLINE    (1u << 3u)
#define FT_BH_NOC_PACKET_RESPONSE_MARKED (1u << 4u)
#define FT_BH_NOC_PACKET_BROADCAST       (1u << 5u)
#define FT_BH_NOC_PACKET_STATIC_VC_ENABLE (1u << 7u)
#define FT_BH_NOC_PACKET_PATH_RESERVE    (1u << 8u)
#define FT_BH_NOC_PACKET_STATIC_VC(vc)   ((vc) << 13u)

/* Atomic opcode and operands in NOC_AT_LEN_BE (BlackholeA0/NoC/Atomics.md, "Atomic increment";
 * layout Diagrams/Src/Bits32.lua NOC_AT_LEN_BE_Increment): opcode at bit 12 (increment = 1),
 * IntWidth ("wrap") at bit 2, Ofs (32-bit word index) at bit 0. The source-index field at bit
 * 10 is not in the documented increment layout: Foundry-defined. */
#define FT_BH_NOC_ATOMIC_OP_INCREMENT_GET     1u
#define FT_BH_NOC_ATOMIC_OPCODE(op)           ((op) << 12u)
#define FT_BH_NOC_ATOMIC_WRAP(width)          ((width) << 2u)
#define FT_BH_NOC_ATOMIC_WORD_INDEX(index)    ((index) << 0u)
#define FT_BH_NOC_ATOMIC_SOURCE_INDEX(index)  ((index) << 10u)

/* Local addresses at or above this are registers, not memory (BlackholeA0/TensixTile/
 * BabyRISCV/README.md memory map: L1 ends at 0x0017FFFF, MMIO starts in the 0xFFxx_xxxx range). */
#define FT_BH_LOCAL_REG_SPACE_BASE 0xFF000000u

/* ---------------------------------------------------------------------------------------------
 * 64-bit "NoC address" packing used by kernels (BlackholeA0/NoC/Coordinates.md, MemoryMap.md
 * NOC_TARG_ADDR_HI layout: 6-bit X then 6-bit Y). A local address of 36 bits sits under the
 * coordinates; the HIGH register receives (noc_addr >> 32) & 0xFFFFFF, i.e. the 4 address bits
 * above bit 32 plus X and Y. The PCIe mask keeps bit 28 of that word (address bit 60) plus the
 * same 4 address bits.
 * ------------------------------------------------------------------------------------------- */

#define FT_BH_NOC_LOCAL_ADDR_BITS     36u
#define FT_BH_NOC_NODE_ID_BITS        6u
#define FT_BH_NOC_COORDINATE_SHIFT    FT_BH_NOC_LOCAL_ADDR_BITS
#define FT_BH_NOC_COORDINATE_MASK     0xFFFFFFu
#define FT_BH_NOC_PCIE_HIGH_ADDR_MASK 0x1000000Fu
#define FT_BH_NOC_NODE_ID_MASK        ((1u << FT_BH_NOC_NODE_ID_BITS) - 1u)

#define FT_BH_NOC_XY_ADDR(x, y, addr)                                                     \
    (((uint64_t)(y) << (FT_BH_NOC_LOCAL_ADDR_BITS + FT_BH_NOC_NODE_ID_BITS)) |             \
     ((uint64_t)(x) << FT_BH_NOC_LOCAL_ADDR_BITS) | (uint64_t)(addr))

/* Broadcast rectangle: end X/Y in the low coordinate slots, start X/Y in the next two
 * (MemoryMap.md, broadcast NOC_TARG_ADDR_HI layout). */
#define FT_BH_NOC_MULTICAST_ADDR(x_start, y_start, x_end, y_end, addr)                    \
    (((uint64_t)(x_start) << (FT_BH_NOC_LOCAL_ADDR_BITS + 2u * FT_BH_NOC_NODE_ID_BITS)) |  \
     ((uint64_t)(y_start) << (FT_BH_NOC_LOCAL_ADDR_BITS + 3u * FT_BH_NOC_NODE_ID_BITS)) |  \
     ((uint64_t)(x_end) << FT_BH_NOC_LOCAL_ADDR_BITS) |                                     \
     ((uint64_t)(y_end) << (FT_BH_NOC_LOCAL_ADDR_BITS + FT_BH_NOC_NODE_ID_BITS)) |          \
     (uint64_t)(addr))

/* Alignment (WormholeB0/NoC/Alignment.md: L1 transfers in 16-byte units, MMIO 4-byte). */
#define FT_BH_NOC_L1_ALIGNMENT_BYTES  16u
#define FT_BH_NOC_REG_ALIGNMENT_BYTES 4u

/* Largest single request: 256 flits of 64 bytes (BlackholeA0/NoC/README.md: 512-bit flits;
 * MemoryMap.md NOC_AT_LEN_BE: at most 16384 bytes per request). */
#define FT_BH_NOC_MAX_BURST_BYTES 16384u

/* Foundry-defined virtual channels. Bits 13..15 of NOC_CTRL are the buddy bit and class bits
 * (MemoryMap.md): 1 = unicast class 0 buddy 1; 4 = class 0b10, required for multicast. */
#define FT_BH_NOC_UNICAST_WRITE_VC   1u
#define FT_BH_NOC_MULTICAST_WRITE_VC 4u

/* Foundry-defined firmware L1 layout: inline-write staging area, 16 bytes per NoC per
 * data-movement processor. */
#define FT_BH_L1_INLINE_STAGING_BASE    32u
#define FT_BH_L1_INLINE_STAGING_PER_NOC 16u

/* ---------------------------------------------------------------------------------------------
 * Per-processor selection. The build defines exactly one COMPILE_FOR_* flavour. The two
 * data-movement cores share the same Foundry-defined command-buffer roles; the compute cores
 * (TRISC) have none, so any NoC command-buffer use there fails to compile.
 * ------------------------------------------------------------------------------------------- */

#if (defined(COMPILE_FOR_BRISC) + defined(COMPILE_FOR_NCRISC) + defined(COMPILE_FOR_TRISC)) != 1
#error "define exactly one of COMPILE_FOR_BRISC, COMPILE_FOR_NCRISC, COMPILE_FOR_TRISC"
#endif

#if defined(COMPILE_FOR_BRISC)
#define FT_BH_PROCESSOR_TYPE 0u
#elif defined(COMPILE_FOR_NCRISC)
#define FT_BH_PROCESSOR_TYPE 1u
#endif

#if !defined(COMPILE_FOR_TRISC)
#ifndef NOC_INDEX
#error "NOC_INDEX must be defined by the build"
#endif
#ifndef NUM_NOCS
#error "NUM_NOCS must be defined by the build"
#endif
#define FT_BH_WRITE_CMD_BUF     0u
#define FT_BH_READ_CMD_BUF      1u
#define FT_BH_REG_WRITE_CMD_BUF 2u
#define FT_BH_ATOMIC_CMD_BUF    3u
#endif

/* ---------------------------------------------------------------------------------------------
 * NoC overlay stream registers (BlackholeA0/TensixTile/BabyRISCV/README.md: NOC_OVERLAY_START_ADDR
 * 0xFFB40000; WormholeB0/NoC/Overlay/AsGeneralUse.md for the registers reused as general MMIO).
 * Each stream owns a 0x1000-byte window of 32-bit registers. The two register indices below are
 * the Foundry CB protocol's choice of general-use registers (the docs name them but give no
 * index numbers).
 * ------------------------------------------------------------------------------------------- */

#define FT_BH_STREAM_REG_BASE        0xFFB40000u
#define FT_BH_STREAM_REG_WINDOW_SIZE 0x1000u
#define FT_BH_STREAM_REG_INDEX_SHIFT 2u

#define FT_BH_STREAM_REG_ADDR(stream, index)                                              \
    (FT_BH_STREAM_REG_BASE + (uint32_t)(stream) * FT_BH_STREAM_REG_WINDOW_SIZE +            \
     ((uint32_t)(index) << FT_BH_STREAM_REG_INDEX_SHIFT))

/* Foundry CB protocol: pages acknowledged by the consumer, and pages received. */
#define FT_BH_STREAM_PAGES_ACKED_INDEX    8u
#define FT_BH_STREAM_PAGES_RECEIVED_INDEX 10u

/* ---------------------------------------------------------------------------------------------
 * Wall clock, low 32 bits, read-only (WormholeB0/EthernetTile/CycleCounters.md for the offset
 * 0x1F0; BlackholeA0/TensixTile/TileControlDebugStatus.md for the 0xFFB12000 block). The soft
 * reset register (BlackholeA0/TensixTile/SoftReset.md, 0xFFB121B0) is in the same block: never
 * write its backend bits from kernel code.
 * ------------------------------------------------------------------------------------------- */

#define FT_BH_WALL_CLOCK_LOW_ADDR 0xFFB121F0u

/* ---------------------------------------------------------------------------------------------
 * Backend configuration fields (BlackholeA0/TensixTile/TensixCoprocessor/BackendConfiguration.md
 * for the space; every value below from the cfg register table). For a field F:
 * FT_BH_CFG_F_WORD is the 32-bit word index, _SHIFT the low bit, _MASK the in-word mask.
 * ------------------------------------------------------------------------------------------- */

/* Number of 128-bit configuration rows per bank (cfg register table). */
#define FT_BH_CFG_STATE_SIZE 56

/* Per-thread config bank select (cfg register table). */
#define FT_BH_CFG_CONFIG_STATE_SELECT_WORD 0
#define FT_BH_CFG_CONFIG_STATE_SELECT_SHIFT 0
#define FT_BH_CFG_CONFIG_STATE_SELECT_MASK 0x1

/* Math destination and number modes (cfg register table). */
#define FT_BH_CFG_DEST_FP32_MODE_WORD 1
#define FT_BH_CFG_DEST_FP32_MODE_SHIFT 29
#define FT_BH_CFG_DEST_FP32_MODE_MASK 0x20000000
#define FT_BH_CFG_SFPU_DEST_FP32_MODE_WORD 1
#define FT_BH_CFG_SFPU_DEST_FP32_MODE_SHIFT 30
#define FT_BH_CFG_SFPU_DEST_FP32_MODE_MASK 0x40000000
#define FT_BH_CFG_INT8_MATH_MODE_WORD 1
#define FT_BH_CFG_INT8_MATH_MODE_SHIFT 31
#define FT_BH_CFG_INT8_MATH_MODE_MASK 0x80000000
#define FT_BH_CFG_SRC_ZERO_FLAG_DISABLE_WORD 2
#define FT_BH_CFG_SRC_ZERO_FLAG_DISABLE_SHIFT 0
#define FT_BH_CFG_SRC_ZERO_FLAG_DISABLE_MASK 0x1
#define FT_BH_CFG_ZEROACC_ABSOLUTE_TILE_MODE_WORD 220
#define FT_BH_CFG_ZEROACC_ABSOLUTE_TILE_MODE_SHIFT 3
#define FT_BH_CFG_ZEROACC_ABSOLUTE_TILE_MODE_MASK 0x8
#define FT_BH_CFG_MATH_DEST_BASE_OFFSET_WORD 1
#define FT_BH_CFG_MATH_DEST_BASE_OFFSET_SHIFT 0
#define FT_BH_CFG_MATH_DEST_BASE_OFFSET_MASK 0xfff
#define FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_WORD 7
#define FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_SHIFT 0
#define FT_BH_CFG_SRCA_DVALID_CLEAR_DISABLE_MASK 0x1

/* Math operand and destination data formats, and their per-thread overrides
 * (cfg register table). */
#define FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_VALUE_WORD 0
#define FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_VALUE_SHIFT 0
#define FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_VALUE_MASK 0xf
#define FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_ENABLE_WORD 0
#define FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_ENABLE_SHIFT 4
#define FT_BH_CFG_MATH_SRCA_FORMAT_OVERRIDE_ENABLE_MASK 0x10
#define FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_VALUE_WORD 0
#define FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_VALUE_SHIFT 5
#define FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_VALUE_MASK 0x1e0
#define FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_ENABLE_WORD 0
#define FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_ENABLE_SHIFT 9
#define FT_BH_CFG_MATH_SRCB_FORMAT_OVERRIDE_ENABLE_MASK 0x200
#define FT_BH_CFG_DEST_FORMAT_OVERRIDE_VALUE_WORD 0
#define FT_BH_CFG_DEST_FORMAT_OVERRIDE_VALUE_SHIFT 10
#define FT_BH_CFG_DEST_FORMAT_OVERRIDE_VALUE_MASK 0x3c00
#define FT_BH_CFG_DEST_FORMAT_OVERRIDE_ENABLE_WORD 0
#define FT_BH_CFG_DEST_FORMAT_OVERRIDE_ENABLE_SHIFT 14
#define FT_BH_CFG_DEST_FORMAT_OVERRIDE_ENABLE_MASK 0x4000
#define FT_BH_CFG_MATH_SRCA_FORMAT_WORD 1
#define FT_BH_CFG_MATH_SRCA_FORMAT_SHIFT 17
#define FT_BH_CFG_MATH_SRCA_FORMAT_MASK 0x1e0000
#define FT_BH_CFG_MATH_SRCA_UNSIGNED_WORD 1
#define FT_BH_CFG_MATH_SRCA_UNSIGNED_SHIFT 15
#define FT_BH_CFG_MATH_SRCA_UNSIGNED_MASK 0x8000
#define FT_BH_CFG_MATH_SRCB_UNSIGNED_WORD 1
#define FT_BH_CFG_MATH_SRCB_UNSIGNED_SHIFT 16
#define FT_BH_CFG_MATH_SRCB_UNSIGNED_MASK 0x10000
#define FT_BH_CFG_DEST_ACCUMULATOR_FORMAT_WORD 1
#define FT_BH_CFG_DEST_ACCUMULATOR_FORMAT_SHIFT 25
#define FT_BH_CFG_DEST_ACCUMULATOR_FORMAT_MASK 0x1e000000

/* Stochastic rounding enables (cfg register table). */
#define FT_BH_CFG_FPU_STOCHASTIC_ROUNDING_WORD 1
#define FT_BH_CFG_FPU_STOCHASTIC_ROUNDING_SHIFT 0
#define FT_BH_CFG_FPU_STOCHASTIC_ROUNDING_MASK 0x1
#define FT_BH_CFG_GASKET_STOCHASTIC_ROUNDING_WORD 1
#define FT_BH_CFG_GASKET_STOCHASTIC_ROUNDING_SHIFT 1
#define FT_BH_CFG_GASKET_STOCHASTIC_ROUNDING_MASK 0x2
#define FT_BH_CFG_PACKER_STOCHASTIC_ROUNDING_WORD 1
#define FT_BH_CFG_PACKER_STOCHASTIC_ROUNDING_SHIFT 2
#define FT_BH_CFG_PACKER_STOCHASTIC_ROUNDING_MASK 0x4

/* Math address modifiers 0..7: SrcA, Dst and bias increments (cfg register table). One word
 * per modifier and per increment kind. */
#define FT_BH_CFG_ADDRMOD0_SRCA_INCREMENT_WORD 12
#define FT_BH_CFG_ADDRMOD0_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD0_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD1_SRCA_INCREMENT_WORD 13
#define FT_BH_CFG_ADDRMOD1_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD1_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD2_SRCA_INCREMENT_WORD 14
#define FT_BH_CFG_ADDRMOD2_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD2_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD3_SRCA_INCREMENT_WORD 15
#define FT_BH_CFG_ADDRMOD3_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD3_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD4_SRCA_INCREMENT_WORD 16
#define FT_BH_CFG_ADDRMOD4_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD4_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD5_SRCA_INCREMENT_WORD 17
#define FT_BH_CFG_ADDRMOD5_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD5_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD6_SRCA_INCREMENT_WORD 18
#define FT_BH_CFG_ADDRMOD6_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD6_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD7_SRCA_INCREMENT_WORD 19
#define FT_BH_CFG_ADDRMOD7_SRCA_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD7_SRCA_INCREMENT_MASK 0x3f
#define FT_BH_CFG_ADDRMOD0_DEST_INCREMENT_WORD 28
#define FT_BH_CFG_ADDRMOD0_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD0_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD1_DEST_INCREMENT_WORD 29
#define FT_BH_CFG_ADDRMOD1_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD1_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD2_DEST_INCREMENT_WORD 30
#define FT_BH_CFG_ADDRMOD2_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD2_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD3_DEST_INCREMENT_WORD 31
#define FT_BH_CFG_ADDRMOD3_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD3_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD4_DEST_INCREMENT_WORD 32
#define FT_BH_CFG_ADDRMOD4_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD4_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD5_DEST_INCREMENT_WORD 33
#define FT_BH_CFG_ADDRMOD5_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD5_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD6_DEST_INCREMENT_WORD 34
#define FT_BH_CFG_ADDRMOD6_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD6_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD7_DEST_INCREMENT_WORD 35
#define FT_BH_CFG_ADDRMOD7_DEST_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD7_DEST_INCREMENT_MASK 0x3ff
#define FT_BH_CFG_ADDRMOD0_BIAS_INCREMENT_WORD 47
#define FT_BH_CFG_ADDRMOD0_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD0_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD1_BIAS_INCREMENT_WORD 48
#define FT_BH_CFG_ADDRMOD1_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD1_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD2_BIAS_INCREMENT_WORD 49
#define FT_BH_CFG_ADDRMOD2_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD2_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD3_BIAS_INCREMENT_WORD 50
#define FT_BH_CFG_ADDRMOD3_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD3_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD4_BIAS_INCREMENT_WORD 51
#define FT_BH_CFG_ADDRMOD4_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD4_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD5_BIAS_INCREMENT_WORD 52
#define FT_BH_CFG_ADDRMOD5_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD5_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD6_BIAS_INCREMENT_WORD 53
#define FT_BH_CFG_ADDRMOD6_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD6_BIAS_INCREMENT_MASK 0xf
#define FT_BH_CFG_ADDRMOD7_BIAS_INCREMENT_WORD 54
#define FT_BH_CFG_ADDRMOD7_BIAS_INCREMENT_SHIFT 0
#define FT_BH_CFG_ADDRMOD7_BIAS_INCREMENT_MASK 0xf

/* Packer address modifiers 0..3: source Y increment (cfg register table). */
#define FT_BH_CFG_PACK_ADDRMOD0_Y_SOURCE_INCREMENT_WORD 37
#define FT_BH_CFG_PACK_ADDRMOD0_Y_SOURCE_INCREMENT_SHIFT 0
#define FT_BH_CFG_PACK_ADDRMOD0_Y_SOURCE_INCREMENT_MASK 0xf
#define FT_BH_CFG_PACK_ADDRMOD1_Y_SOURCE_INCREMENT_WORD 38
#define FT_BH_CFG_PACK_ADDRMOD1_Y_SOURCE_INCREMENT_SHIFT 0
#define FT_BH_CFG_PACK_ADDRMOD1_Y_SOURCE_INCREMENT_MASK 0xf
#define FT_BH_CFG_PACK_ADDRMOD2_Y_SOURCE_INCREMENT_WORD 39
#define FT_BH_CFG_PACK_ADDRMOD2_Y_SOURCE_INCREMENT_SHIFT 0
#define FT_BH_CFG_PACK_ADDRMOD2_Y_SOURCE_INCREMENT_MASK 0xf
#define FT_BH_CFG_PACK_ADDRMOD3_Y_SOURCE_INCREMENT_WORD 40
#define FT_BH_CFG_PACK_ADDRMOD3_Y_SOURCE_INCREMENT_SHIFT 0
#define FT_BH_CFG_PACK_ADDRMOD3_Y_SOURCE_INCREMENT_MASK 0xf

/* Unpacker tile descriptors and per-unpacker formats (cfg register table). The descriptor is
 * 128 bits wide, so it has no 32-bit mask. */
#define FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_WORD 64
#define FT_BH_CFG_UNPACKER0_TILE_DESCRIPTOR_SHIFT 0
#define FT_BH_CFG_UNPACKER1_TILE_DESCRIPTOR_WORD 112
#define FT_BH_CFG_UNPACKER1_TILE_DESCRIPTOR_SHIFT 0
#define FT_BH_CFG_UNPACKER0_LF8_4BIT_EXPONENT_WORD 71
#define FT_BH_CFG_UNPACKER0_LF8_4BIT_EXPONENT_SHIFT 22
#define FT_BH_CFG_UNPACKER0_LF8_4BIT_EXPONENT_MASK 0x400000
#define FT_BH_CFG_UNPACKER1_LF8_4BIT_EXPONENT_WORD 119
#define FT_BH_CFG_UNPACKER1_LF8_4BIT_EXPONENT_SHIFT 22
#define FT_BH_CFG_UNPACKER1_LF8_4BIT_EXPONENT_MASK 0x400000
#define FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_WORD 72
#define FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_SHIFT 0
#define FT_BH_CFG_UNPACKER0_OUTPUT_FORMAT_MASK 0xf
#define FT_BH_CFG_UNPACKER1_OUTPUT_FORMAT_WORD 120
#define FT_BH_CFG_UNPACKER1_OUTPUT_FORMAT_SHIFT 0
#define FT_BH_CFG_UNPACKER1_OUTPUT_FORMAT_MASK 0xf
#define FT_BH_CFG_UNPACKER0_HALOIZE_MODE_WORD 72
#define FT_BH_CFG_UNPACKER0_HALOIZE_MODE_SHIFT 8
#define FT_BH_CFG_UNPACKER0_HALOIZE_MODE_MASK 0x100

/* Unpacker L1 base addresses per config context, and context 0 destination layout
 * (cfg register table). */
#define FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_WORD 76
#define FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_SHIFT 0
#define FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT0_MASK 0xffffffff
#define FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_WORD 77
#define FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_SHIFT 0
#define FT_BH_CFG_UNPACKER0_L1_BASE_CONTEXT1_MASK 0xffffffff
#define FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_WORD 124
#define FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_SHIFT 0
#define FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT0_MASK 0xffffffff
#define FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_WORD 125
#define FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_SHIFT 0
#define FT_BH_CFG_UNPACKER1_L1_BASE_CONTEXT1_MASK 0xffffffff
#define FT_BH_CFG_UNPACKER0_DEST_CONTEXT0_ADDRESS_WORD 84
#define FT_BH_CFG_UNPACKER0_DEST_CONTEXT0_ADDRESS_SHIFT 0
#define FT_BH_CFG_UNPACKER0_DEST_CONTEXT0_ADDRESS_MASK 0xffff
#define FT_BH_CFG_UNPACKER0_CONTEXT0_TILE_X_DIM_WORD 86
#define FT_BH_CFG_UNPACKER0_CONTEXT0_TILE_X_DIM_SHIFT 0
#define FT_BH_CFG_UNPACKER0_CONTEXT0_TILE_X_DIM_MASK 0xffff
#define FT_BH_CFG_UNPACKER0_ADD_DEST_ADDRESS_COUNTER_WORD 50
#define FT_BH_CFG_UNPACKER0_ADD_DEST_ADDRESS_COUNTER_SHIFT 8
#define FT_BH_CFG_UNPACKER0_ADD_DEST_ADDRESS_COUNTER_MASK 0x100
#define FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_WORD 41
#define FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_SHIFT 0
#define FT_BH_CFG_UNPACKER0_CONTEXT_OFFSET_MASK 0xf
#define FT_BH_CFG_SRCA_WRITE_BASE_WORD 5
#define FT_BH_CFG_SRCA_WRITE_BASE_SHIFT 0
#define FT_BH_CFG_SRCA_WRITE_BASE_MASK 0x3

/* Unpacker address-generator strides: input (L1 side) and output (register side), per
 * WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md; values cfg register table. */
#define FT_BH_CFG_UNPACKER0_INPUT_Y_STRIDE_WORD 44
#define FT_BH_CFG_UNPACKER0_INPUT_Y_STRIDE_SHIFT 16
#define FT_BH_CFG_UNPACKER0_INPUT_Y_STRIDE_MASK 0xffff0000
#define FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_WORD 56
#define FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_SHIFT 16
#define FT_BH_CFG_UNPACKER0_OUTPUT_Y_STRIDE_MASK 0xffff0000
#define FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_WORD 57
#define FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_SHIFT 0
#define FT_BH_CFG_UNPACKER0_OUTPUT_Z_STRIDE_MASK 0xffff
#define FT_BH_CFG_UNPACKER0_OUTPUT_W_STRIDE_WORD 57
#define FT_BH_CFG_UNPACKER0_OUTPUT_W_STRIDE_SHIFT 16
#define FT_BH_CFG_UNPACKER0_OUTPUT_W_STRIDE_MASK 0xffff0000
#define FT_BH_CFG_UNPACKER1_OUTPUT_Z_STRIDE_WORD 59
#define FT_BH_CFG_UNPACKER1_OUTPUT_Z_STRIDE_SHIFT 0
#define FT_BH_CFG_UNPACKER1_OUTPUT_Z_STRIDE_MASK 0xffff
#define FT_BH_CFG_UNPACKER1_OUTPUT_W_STRIDE_WORD 59
#define FT_BH_CFG_UNPACKER1_OUTPUT_W_STRIDE_SHIFT 16
#define FT_BH_CFG_UNPACKER1_OUTPUT_W_STRIDE_MASK 0xffff0000

/* Packer 0 input strides (WormholeB0/TensixTile/TensixCoprocessor/Packers/
 * InputAddressGenerator.md); values cfg register table. */
#define FT_BH_CFG_PACKER0_INPUT_X_STRIDE_WORD 12
#define FT_BH_CFG_PACKER0_INPUT_X_STRIDE_SHIFT 0
#define FT_BH_CFG_PACKER0_INPUT_X_STRIDE_MASK 0xffff
#define FT_BH_CFG_PACKER0_INPUT_Y_STRIDE_WORD 12
#define FT_BH_CFG_PACKER0_INPUT_Y_STRIDE_SHIFT 16
#define FT_BH_CFG_PACKER0_INPUT_Y_STRIDE_MASK 0xffff0000
#define FT_BH_CFG_PACKER0_INPUT_Z_STRIDE_WORD 13
#define FT_BH_CFG_PACKER0_INPUT_Z_STRIDE_SHIFT 0
#define FT_BH_CFG_PACKER0_INPUT_Z_STRIDE_MASK 0xffff
#define FT_BH_CFG_PACKER0_INPUT_W_STRIDE_WORD 13
#define FT_BH_CFG_PACKER0_INPUT_W_STRIDE_SHIFT 16
#define FT_BH_CFG_PACKER0_INPUT_W_STRIDE_MASK 0xffff0000

/* Packer 0 output format, exponent handling, L1 destination and Dst read controls
 * (cfg register table). */
#define FT_BH_CFG_PACKER0_LF8_4BIT_EXPONENT_WORD 71
#define FT_BH_CFG_PACKER0_LF8_4BIT_EXPONENT_SHIFT 23
#define FT_BH_CFG_PACKER0_LF8_4BIT_EXPONENT_MASK 0x800000
#define FT_BH_CFG_PACKER0_ROW_START_SECTION_SIZE_WORD 68
#define FT_BH_CFG_PACKER0_ROW_START_SECTION_SIZE_SHIFT 0
#define FT_BH_CFG_PACKER0_ROW_START_SECTION_SIZE_MASK 0xffff
#define FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_ENABLE_WORD 71
#define FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_ENABLE_SHIFT 20
#define FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_ENABLE_MASK 0x100000
#define FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_WORD 71
#define FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_SHIFT 24
#define FT_BH_CFG_PACKER0_EXPONENT_THRESHOLD_MASK 0xff000000
#define FT_BH_CFG_PACKER0_ZERO_FLAGS_DISABLE_WORD 70
#define FT_BH_CFG_PACKER0_ZERO_FLAGS_DISABLE_SHIFT 2
#define FT_BH_CFG_PACKER0_ZERO_FLAGS_DISABLE_MASK 0x4
#define FT_BH_CFG_PACKER0_L1_ACCUMULATE_WORD 71
#define FT_BH_CFG_PACKER0_L1_ACCUMULATE_SHIFT 19
#define FT_BH_CFG_PACKER0_L1_ACCUMULATE_MASK 0x80000
#define FT_BH_CFG_PACKER0_L1_DEST_ADDRESS_WORD 69
#define FT_BH_CFG_PACKER0_L1_DEST_ADDRESS_SHIFT 0
#define FT_BH_CFG_PACKER0_L1_DEST_ADDRESS_MASK 0xffffffff
#define FT_BH_CFG_PACKER0_EXPONENT_SECTION_SIZE_WORD 96
#define FT_BH_CFG_PACKER0_EXPONENT_SECTION_SIZE_SHIFT 16
#define FT_BH_CFG_PACKER0_EXPONENT_SECTION_SIZE_MASK 0xffff0000
#define FT_BH_CFG_PACKER_DEST_READ_32BIT_WORD 18
#define FT_BH_CFG_PACKER_DEST_READ_32BIT_SHIFT 0
#define FT_BH_CFG_PACKER_DEST_READ_32BIT_MASK 0x1
#define FT_BH_CFG_PACKER0_DATUMS_PER_XY_PLANE_WORD 28
#define FT_BH_CFG_PACKER0_DATUMS_PER_XY_PLANE_SHIFT 0
#define FT_BH_CFG_PACKER0_DATUMS_PER_XY_PLANE_MASK 0xff
#define FT_BH_CFG_PACKER0_READS_PER_XY_PLANE_WORD 28
#define FT_BH_CFG_PACKER0_READS_PER_XY_PLANE_SHIFT 8
#define FT_BH_CFG_PACKER0_READS_PER_XY_PLANE_MASK 0xff00
#define FT_BH_CFG_PACKER0_DEST_BASE_OFFSET_WORD 180
#define FT_BH_CFG_PACKER0_DEST_BASE_OFFSET_SHIFT 0
#define FT_BH_CFG_PACKER0_DEST_BASE_OFFSET_MASK 0xfff

/* Packer edge masks and tile row-set mapping (WormholeB0/TensixTile/TensixCoprocessor/
 * Packers/EdgeMasking.md), and ReLU (Packers/ReLU.md); values cfg register table. */
#define FT_BH_CFG_PACKER_EDGE_MASK0_WORD 24
#define FT_BH_CFG_PACKER_EDGE_MASK0_SHIFT 0
#define FT_BH_CFG_PACKER_EDGE_MASK0_MASK 0xffff
#define FT_BH_CFG_PACKER_EDGE_MASK1_WORD 25
#define FT_BH_CFG_PACKER_EDGE_MASK1_SHIFT 0
#define FT_BH_CFG_PACKER_EDGE_MASK1_MASK 0xffff
#define FT_BH_CFG_PACKER_TILE_ROW_MAP0_WORD 20
#define FT_BH_CFG_PACKER_TILE_ROW_MAP0_SHIFT 0
#define FT_BH_CFG_PACKER_TILE_ROW_MAP0_MASK 0x3
#define FT_BH_CFG_PACKER_TILE_ROW_MAP1_WORD 21
#define FT_BH_CFG_PACKER_TILE_ROW_MAP1_SHIFT 0
#define FT_BH_CFG_PACKER_TILE_ROW_MAP1_MASK 0x3
#define FT_BH_CFG_PACKER_RELU_MODE_WORD 2
#define FT_BH_CFG_PACKER_RELU_MODE_SHIFT 2
#define FT_BH_CFG_PACKER_RELU_MODE_MASK 0x3c
#define FT_BH_CFG_PACKER_RELU_THRESHOLD_WORD 2
#define FT_BH_CFG_PACKER_RELU_THRESHOLD_SHIFT 6
#define FT_BH_CFG_PACKER_RELU_THRESHOLD_MASK 0x3fffc0

/* ---------------------------------------------------------------------------------------------
 * Primitive register accesses. The access shape (one load or store, which pointer qualifiers)
 * is part of the contract: keep each body as written.
 * ------------------------------------------------------------------------------------------- */

/** One 32-bit load from a local register. Thread-safe: n/a (MMIO). Reentrant: yes. */
static FT_BH_INLINE uint32_t ft_bh_reg_read(uint32_t addr)
{
    return *((volatile FT_BH_REG_PTR uint32_t *)addr);
}

/** Drop this core's L0 data cache: it has no maintenance operation, but a fence flushes it
 *  (BlackholeA0/TensixTile/BabyRISCV/InstructionSet.md, MemoryOrdering.md). */
static FT_BH_INLINE void ft_bh_l1_cache_invalidate(void)
{
    __asm__ volatile("fence");
}

/** Low 32 bits of the free-running wall clock. */
static FT_BH_INLINE uint32_t ft_bh_wall_clock_low(void)
{
    return ft_bh_reg_read(FT_BH_WALL_CLOCK_LOW_ADDR);
}

/** Store one initiator register. `reg` is an absolute NoC-0 initiator-0 register address
 *  (one of the FT_BH_NOC_REG_* macros); `cmd_buf` and `noc` select the initiator and NIU. */
static FT_BH_INLINE void ft_bh_noc_cmd_write(uint32_t noc, uint32_t cmd_buf, uint32_t reg,
                                             uint32_t value)
{
    uint32_t addr = (cmd_buf << FT_BH_NOC_CMD_BUF_SELECT_BIT) +
                    (noc << FT_BH_NOC_INSTANCE_SELECT_BIT) + reg;
    *((volatile uint32_t *)addr) = value;
}

/** Load one initiator register; addressing as ft_bh_noc_cmd_write. */
static FT_BH_INLINE uint32_t ft_bh_noc_cmd_read(uint32_t noc, uint32_t cmd_buf, uint32_t reg)
{
    uint32_t addr = (cmd_buf << FT_BH_NOC_CMD_BUF_SELECT_BIT) +
                    (noc << FT_BH_NOC_INSTANCE_SELECT_BIT) + reg;
    return *((volatile uint32_t *)addr);
}

/** Load one NIU status counter of `noc`. */
static FT_BH_INLINE uint32_t ft_bh_noc_status_read(uint32_t noc, uint32_t counter)
{
    uint32_t addr = (noc << FT_BH_NOC_INSTANCE_SELECT_BIT) + FT_BH_NOC_STATUS_ADDR(counter);
    return *((volatile uint32_t *)addr);
}

/** Nonzero when the initiator has taken its last request and may be reprogrammed. */
static FT_BH_INLINE int ft_bh_noc_cmd_buf_ready(uint32_t noc, uint32_t cmd_buf)
{
    return ft_bh_noc_cmd_read(noc, cmd_buf, FT_BH_NOC_REG_COMMAND_CONTROL) ==
           FT_BH_NOC_COMMAND_STATUS_READY;
}

#endif /* FT_BH_REGS_H */
