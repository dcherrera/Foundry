/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_memmap.h
 * @brief Foundry's L1 map for one Blackhole Tensix tile, plus the few hardware addresses and
 *   sizes the host and Foundry's firmware both depend on. Shared by the host library (tt/),
 *   Foundry's firmware (tt/fw/), the kernels (tt/kernels/) and the generated linker scripts.
 *
 * TWO KINDS OF VALUE live here, and each one says which it is:
 *   [HW]      fixed by the silicon; cited to tt-isa-documentation BlackholeA0 (the page named).
 *   [FOUNDRY] Foundry's own choice. Both ends of every [FOUNDRY] region are Foundry code, so the
 *             value is ours to define. Each one equals the value the shipped images were linked
 *             against, which is what keeps every existing image and the host byte-identical.
 *
 * THE L1 MAP is one list of regions in address order, each with an explicit size. A region
 * Foundry does not use is one named reserved span, so every used address keeps its value.
 * Where both neighbours are Foundry's, the next base is computed (base + size); otherwise the
 * address is stated.
 *
 * EVERY OBJECT-LIKE MACRO IS A PLAIN INTEGER EXPRESSION. tools/emit_tensix_linker_script.py
 * reads them through `cc -E -dM` and evaluates them itself: no casts, no sizeof, no offsetof,
 * no division, no conditional operator. The function-like helpers are exempt.
 *
 * @status Clean-room rewrite (plan 4.3), from the public docs, Foundry's own consumers and the
 *   measured layout of the previous build. Every value consumers use is unchanged.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_MEMMAP_H
#define FT_TT_MEMMAP_H

#include <stdint.h>

/* ================================================================
 * Compile-time assertion, usable at file scope from C99 host code and from the firmware
 * ================================================================ */

/** Fails the build when `condition` is false. `tag` must be unique within a translation unit. */
#define FT_TT_STATIC_ASSERT(condition, tag) \
    typedef char ft_tt_static_assert_##tag[(condition) ? 1 : -1]

/* ================================================================
 * Memories outside L1
 * ================================================================ */

/** [HW] 1536 KiB of L1 per Tensix tile (TensixTile/README.md; BabyRISCV/README.md memory map,
 *  MEM_L1_BASE 0x0000_0000..0x0017_FFFF). */
#define FT_TT_MEM_L1_SIZE (1536u * 1024u)

/** [FOUNDRY] Bytes of one DRAM bank Foundry addresses: 16 MiB short of the 4 GiB each bank
 *  exposes (NoC/Coordinates.md). Checked against the arch table's dram_bytes_per_bank. */
#define FT_TT_MEM_DRAM_SIZE 0xFF000000u

/** [FOUNDRY] DRAM banks Foundry interleaves over. */
#define FT_TT_NUM_DRAM_BANKS 8u

/** [HW] Each RISC-V's private data RAM (BabyRISCV/README.md "Local Data RAM", MEM_LOCAL_BASE):
 *  8 KiB on B and NC, 4 KiB on T0, T1, T2. */
#define FT_TT_MEM_LOCAL_BASE        0xFFB00000u
#define FT_TT_MEM_BRISC_LOCAL_SIZE  (8u * 1024u)
#define FT_TT_MEM_NCRISC_LOCAL_SIZE (8u * 1024u)
#define FT_TT_MEM_TRISC_LOCAL_SIZE  (4u * 1024u)

/* ================================================================
 * The L1 map, in address order
 * ================================================================ */

/* 0x0000: BRISC's reset PC is the fixed value 0 (SoftReset.md). Foundry does not use the word
 * there. */

/** [FOUNDRY] Return-value word for NoC atomics. BRISC firmware presets every NoC's atomic
 *  initiator to return here (fw/ft_brisc_fw.c). */
#define FT_TT_MEM_NOC_ATOMIC_RET_VAL_ADDR 0x4u
#define FT_TT_MEM_NOC_ATOMIC_RET_VAL_SIZE 4u

/** [FOUNDRY] Reserved: Foundry places nothing between the atomic word and the mailbox. */
#define FT_TT_MEM_LOW_RESERVED_BASE \
    (FT_TT_MEM_NOC_ATOMIC_RET_VAL_ADDR + FT_TT_MEM_NOC_ATOMIC_RET_VAL_SIZE)
#define FT_TT_MEM_LOW_RESERVED_SIZE 0x58u

/** [FOUNDRY] The host <-> firmware mailbox (ft_tt_mailbox.h). The region is larger than the
 *  mailbox struct; the host clears the whole of [BASE, END) before booting a core. */
#define FT_TT_MEM_MAILBOX_BASE (FT_TT_MEM_LOW_RESERVED_BASE + FT_TT_MEM_LOW_RESERVED_SIZE)
#define FT_TT_MEM_MAILBOX_SIZE 0x33F0u
#define FT_TT_MEM_MAILBOX_END  (FT_TT_MEM_MAILBOX_BASE + FT_TT_MEM_MAILBOX_SIZE)

/** [FOUNDRY] Reserved gap after the mailbox. */
#define FT_TT_MEM_MAILBOX_GAP_SIZE 0x10u

/** [FOUNDRY] A block BRISC firmware zeroes at every boot (fw/ft_brisc_fw.c). */
#define FT_TT_MEM_ZEROS_BASE (FT_TT_MEM_MAILBOX_END + FT_TT_MEM_MAILBOX_GAP_SIZE)
#define FT_TT_MEM_ZEROS_SIZE 0x200u

/** [FOUNDRY] Reserved: not used by Foundry. */
#define FT_TT_MEM_PRE_FIRMWARE_RESERVED_BASE (FT_TT_MEM_ZEROS_BASE + FT_TT_MEM_ZEROS_SIZE)
#define FT_TT_MEM_PRE_FIRMWARE_RESERVED_SIZE 0x400u

/**
 * [FOUNDRY] The five firmware slots, back to back. The host loads each processor's firmware
 * here and BRISC programs the subordinates' reset PCs to these bases (SoftReset.md: the TRISC0/1/2
 * and NCRISC reset PC registers and their overrides); the emitted linker scripts place the
 * images.
 */
#define FT_TT_MEM_BRISC_FIRMWARE_SIZE  0x2200u
#define FT_TT_MEM_NCRISC_FIRMWARE_SIZE 0xA00u
#define FT_TT_MEM_TRISC0_FIRMWARE_SIZE 0xA00u
#define FT_TT_MEM_TRISC1_FIRMWARE_SIZE 0xA00u
#define FT_TT_MEM_TRISC2_FIRMWARE_SIZE 0xA00u

#define FT_TT_MEM_BRISC_FIRMWARE_BASE \
    (FT_TT_MEM_PRE_FIRMWARE_RESERVED_BASE + FT_TT_MEM_PRE_FIRMWARE_RESERVED_SIZE)
#define FT_TT_MEM_NCRISC_FIRMWARE_BASE \
    (FT_TT_MEM_BRISC_FIRMWARE_BASE + FT_TT_MEM_BRISC_FIRMWARE_SIZE)
#define FT_TT_MEM_TRISC0_FIRMWARE_BASE \
    (FT_TT_MEM_NCRISC_FIRMWARE_BASE + FT_TT_MEM_NCRISC_FIRMWARE_SIZE)
#define FT_TT_MEM_TRISC1_FIRMWARE_BASE \
    (FT_TT_MEM_TRISC0_FIRMWARE_BASE + FT_TT_MEM_TRISC0_FIRMWARE_SIZE)
#define FT_TT_MEM_TRISC2_FIRMWARE_BASE \
    (FT_TT_MEM_TRISC1_FIRMWARE_BASE + FT_TT_MEM_TRISC1_FIRMWARE_SIZE)
#define FT_TT_MEM_FIRMWARE_END \
    (FT_TT_MEM_TRISC2_FIRMWARE_BASE + FT_TT_MEM_TRISC2_FIRMWARE_SIZE)

/** [FOUNDRY] Reserved: not used by Foundry. */
#define FT_TT_MEM_POST_FIRMWARE_RESERVED_SIZE 0x100u

/**
 * [FOUNDRY] First byte of the reserved tail of the read-only region (up to
 * FT_TT_MEM_MAP_READ_ONLY_END). Foundry never stores anything there.
 */
#define FT_TT_MEM_TENSIX_ROUTING_TABLE_BASE \
    (FT_TT_MEM_FIRMWARE_END + FT_TT_MEM_POST_FIRMWARE_RESERVED_SIZE)

/** [FOUNDRY] End of the region no kernel may write. */
#define FT_TT_MEM_MAP_READ_ONLY_END 0x9100u

/** [FOUNDRY] End of the map. [READ_ONLY_END, MAP_END) is reserved. */
#define FT_TT_MEM_MAP_END 0x9F80u

/** [FOUNDRY] Everything from the end of the map to the end of L1 is the usable arena. */
#define FT_TT_L1_USABLE_BASE FT_TT_MEM_MAP_END
#define FT_TT_L1_USABLE_SIZE (FT_TT_MEM_L1_SIZE - FT_TT_L1_USABLE_BASE)

/**
 * [FOUNDRY] Kernel-staging scratch, one slot per processor, at the start of the usable arena.
 * Each firmware copies its initialised data from its slot into its local data RAM at boot
 * (fw/ft_*_fw.c initialise_local_memory()), so a slot is as large as that processor's local
 * data RAM, and the slots are live only while a core boots.
 */
#define FT_TT_MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH FT_TT_L1_USABLE_BASE
#define FT_TT_MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH \
    (FT_TT_MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH + FT_TT_MEM_BRISC_LOCAL_SIZE)
#define FT_TT_MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH \
    (FT_TT_MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH + FT_TT_MEM_NCRISC_LOCAL_SIZE)
#define FT_TT_MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH \
    (FT_TT_MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH + FT_TT_MEM_TRISC_LOCAL_SIZE)
#define FT_TT_MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH \
    (FT_TT_MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH + FT_TT_MEM_TRISC_LOCAL_SIZE)
#define FT_TT_MEM_INIT_LOCAL_SCRATCH_END \
    (FT_TT_MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH + FT_TT_MEM_TRISC_LOCAL_SIZE)

/** [FOUNDRY] Reserved span after the staging slots. */
#define FT_TT_MEM_POST_SCRATCH_RESERVED_SIZE 0x2800u

/** [FOUNDRY] A small staging span the L1 layout tests keep clear of. */
#define FT_TT_MEM_LOGICAL_TO_VIRTUAL_SCRATCH \
    (FT_TT_MEM_INIT_LOCAL_SCRATCH_END + FT_TT_MEM_POST_SCRATCH_RESERVED_SIZE)
#define FT_TT_MEM_LOGICAL_TO_VIRTUAL_SIZE 0x20u

/* ================================================================
 * Linker limits per processor
 * ================================================================ */

/** [FOUNDRY] Upper bound the linker scripts put on one kernel image. NOT an allocation bound:
 *  it is larger than the usable arena; ft_tt_buffer.h bounds by FT_TT_MEM_L1_SIZE instead. */
#define FT_TT_MEM_MAX_KERNEL_SIZE    0x176400u
#define FT_TT_MEM_BRISC_KERNEL_SIZE  FT_TT_MEM_MAX_KERNEL_SIZE
#define FT_TT_MEM_NCRISC_KERNEL_SIZE FT_TT_MEM_MAX_KERNEL_SIZE
#define FT_TT_MEM_TRISC0_KERNEL_SIZE FT_TT_MEM_MAX_KERNEL_SIZE
#define FT_TT_MEM_TRISC1_KERNEL_SIZE FT_TT_MEM_MAX_KERNEL_SIZE
#define FT_TT_MEM_TRISC2_KERNEL_SIZE FT_TT_MEM_MAX_KERNEL_SIZE

/** [FOUNDRY] Smallest stack the linker scripts leave below the top of local data RAM. */
#define FT_TT_MEM_BRISC_STACK_MIN_SIZE  256u
#define FT_TT_MEM_NCRISC_STACK_MIN_SIZE 256u
#define FT_TT_MEM_TRISC0_STACK_MIN_SIZE 192u
#define FT_TT_MEM_TRISC1_STACK_MIN_SIZE 192u
#define FT_TT_MEM_TRISC2_STACK_MIN_SIZE 256u

/* ================================================================
 * Grid, alignment, board
 * ================================================================ */

/** [HW] The physical NoC grid is 17 x 12 tiles (NoC/Coordinates.md, the NoC #0 / #1 grid
 *  diagrams). ft_tt_arch_check_headers() holds these to the arch table. */
#define FT_TT_NOC_SIZE_X 17u
#define FT_TT_NOC_SIZE_Y 12u

/** [FOUNDRY] Tensix harvesting removes whole columns: ft_tt_harvest_axis's COLUMN (2) in
 *  ft_tt_arch.h; ft_tt_arch_check_headers() holds the two together. */
#define FT_TT_TENSIX_HARVEST_AXIS 2u

/** [FOUNDRY] Alignment of every NoC transfer's L1 and DRAM addresses. */
#define FT_TT_L1_ALIGNMENT   16u
#define FT_TT_DRAM_ALIGNMENT 64u

/** [FOUNDRY] The board the kernel images are built for (tensix_kernel_build_arch.sh carries
 *  the same pair; ft_tt_coord_check_kernel_build_assumptions() compares them with the card). */
#define FT_TT_KERNEL_BUILD_L1_BANK_COUNT   140u
#define FT_TT_KERNEL_BUILD_DRAM_BANK_COUNT 8u

/* ================================================================
 * Processors, in the order every per-processor array uses
 * ================================================================ */

/** [FOUNDRY] Two data-movement RISC-Vs (B, NC) and three compute RISC-Vs (T0, T1, T2); five
 *  in all (TensixTile/README.md "5x Baby RISCV cores"). */
#define FT_TT_PROC_DM0   0u
#define FT_TT_PROC_DM1   1u
#define FT_TT_PROC_MATH0 2u
#define FT_TT_PROC_MATH1 3u
#define FT_TT_PROC_MATH2 4u
#define FT_TT_PROC_COUNT 5u

/* ================================================================
 * Overlay stream registers (circular-buffer page counters)
 * ================================================================ */

/** [HW] NOC_OVERLAY_START_ADDR (BabyRISCV/README.md memory map). */
#define FT_TT_NOC_OVERLAY_BASE 0xFFB40000u

/** [FOUNDRY] Address stride between consecutive streams' register blocks. */
#define FT_TT_NOC_STREAM_REG_SPACE_SIZE 0x1000u

/** Address of 32-bit register `index` of overlay stream `stream`. */
#define FT_TT_STREAM_REG_ADDR(stream, index)                                         \
    (FT_TT_NOC_OVERLAY_BASE + (uint32_t)(stream) * FT_TT_NOC_STREAM_REG_SPACE_SIZE + \
     ((uint32_t)(index) << 2))

/** [FOUNDRY] Circular buffer `cb` keeps its page counters in stream `cb`: pages received in
 *  register 10, pages acknowledged in register 8. */
#define FT_TT_CB_TILES_RECEIVED_ADDR(cb) FT_TT_STREAM_REG_ADDR((cb), 10u)
#define FT_TT_CB_TILES_ACKED_ADDR(cb)    FT_TT_STREAM_REG_ADDR((cb), 8u)

/* ================================================================
 * The map holds together
 * ================================================================ */

FT_TT_STATIC_ASSERT(FT_TT_PROC_COUNT == 5u, memmap_five_processors);
FT_TT_STATIC_ASSERT(FT_TT_MEM_MAILBOX_BASE == 0x60u, memmap_mailbox_base);
FT_TT_STATIC_ASSERT(FT_TT_MEM_ZEROS_BASE == 0x3460u, memmap_zeros_base);
FT_TT_STATIC_ASSERT(FT_TT_MEM_BRISC_FIRMWARE_BASE == 0x3A60u, memmap_brisc_firmware_base);
FT_TT_STATIC_ASSERT(FT_TT_MEM_FIRMWARE_END == 0x8460u, memmap_firmware_end);
FT_TT_STATIC_ASSERT(FT_TT_MEM_TENSIX_ROUTING_TABLE_BASE < FT_TT_MEM_MAP_READ_ONLY_END,
                    memmap_protected_target_is_read_only);
FT_TT_STATIC_ASSERT(FT_TT_MEM_MAP_READ_ONLY_END <= FT_TT_MEM_MAP_END, memmap_read_only_in_map);
FT_TT_STATIC_ASSERT(FT_TT_MEM_INIT_LOCAL_SCRATCH_END <= FT_TT_MEM_LOGICAL_TO_VIRTUAL_SCRATCH,
                    memmap_staging_slots_in_order);
FT_TT_STATIC_ASSERT(FT_TT_MEM_LOGICAL_TO_VIRTUAL_SCRATCH == 0x13780u, memmap_staging_span);
FT_TT_STATIC_ASSERT(FT_TT_L1_USABLE_BASE % FT_TT_L1_ALIGNMENT == 0u, memmap_arena_aligned);

#endif /* FT_TT_MEMMAP_H */
