/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_phase2_l1_layout.h
 * @brief Where the Phase 2 bring-up tests put things in one Tensix's L1 — DERIVED at run
 *   time from where the firmware images actually landed, not hand-chosen.
 *
 *   One header rather than per-test constants, because the tests have to agree:
 *   test_phase2_images.c checks that no loaded kernel reaches the allocation floor,
 *   test_phase2_launch.c plants runtime args here, and test_phase2_vector_add.c puts the
 *   circular buffers and the data buffers here.
 *
 *   The base used to be the literal 0x14000, picked by hand to clear the firmware `.data`
 *   staging and the bank tables. ft_tt_image_placement_l1_floor() exists to compute
 *   exactly that number, so the base now comes from
 *   ft_tt_phase2_l1_layout_derive(): the highest address the five FIRMWARE images
 *   reached, or the end of the regions the stock memory map reserves above MEM_MAP_END,
 *   whichever is higher. Everything else is a fixed offset from it.
 *
 * Constraints the offsets satisfy, all from docs/tt_launch_notes.md §2:
 *   - the base is at or above FT_TT_L1_USABLE_BASE (0x9F80), the allocator floor;
 *   - and above the firmware `.data` staging, NCRISC's IRAM staging and the two bank
 *     tables, which the usable-base macro does NOT cover — see
 *     FT_TT_PHASE2_RESERVED_L1_END;
 *   - the CB-config array and every runtime-arg block sit within 64 KiB of the base,
 *     because those wire fields are 16-bit;
 *   - the base is 16-byte aligned (FT_TT_MAX_NOC_WRITE_ALIGNMENT), and so is every
 *     offset, because a CB FIFO address is shifted by 4 bits on a TRISC.
 *
 * Kernel TEXT is not in this map. It is linked at a fixed low address just after its
 * processor's firmware, not allocated from here — see the notes' §2.
 *
 * @status Used by the on-card Phase 2 tests.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_PHASE2_L1_LAYOUT_H
#define FT_TT_PHASE2_L1_LAYOUT_H

#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_memmap.h"

/**
 * First L1 address above every region the stock memory map stages something into.
 *
 * FT_TT_L1_USABLE_BASE is MEM_MAP_END, and it is NOT this number: the five per-processor
 * `.data` staging regions, NCRISC's IRAM staging and the two bank tables all live ABOVE
 * it (ft_tt_memmap.h). A floor derived from the loaded firmware images alone would sit
 * under the bank tables, because those are written by firmware and are in no ELF.
 */
#define FT_TT_PHASE2_RESERVED_L1_END \
    (FT_TT_MEM_LOGICAL_TO_VIRTUAL_SCRATCH + FT_TT_MEM_LOGICAL_TO_VIRTUAL_SIZE)

/**
 * What the derived base is rounded up to — the DRAM alignment, 64 B, NOT L1's 16 B.
 *
 * **This cost a debugging session and is the non-obvious part of this file.** A circular
 * buffer in this map is the LOCAL endpoint of the reader's `noc_async_read` from DRAM,
 * and the binding alignment for a transfer is the stricter of the two ends. The
 * hand-chosen 0x14000 satisfied it by accident; the first derived base, 0x137A0, is
 * 16 B aligned and 32 B aligned but not 64, and every tile came back with its LAST 64
 * BYTES wrong — a silent, partial, byte-exact-until-element-992 corruption, not a
 * refusal. Every offset below is a multiple of 64, so aligning the base is enough to
 * align the whole map.
 */
#define FT_TT_PHASE2_LAYOUT_ALIGNMENT FT_TT_DRAM_ALIGNMENT

/* Offsets from the derived base. Kept as macros so the relationship between them is
 * readable in one place; the addresses themselves are struct fields below. */
#define FT_TT_PHASE2_CB_CONFIG_OFFSET  0x0000u /**< 64 CBs x 4 words */
#define FT_TT_PHASE2_RTA_OFFSET        0x0400u
#define FT_TT_PHASE2_RTA_STRIDE        0x0080u
#define FT_TT_PHASE2_PROBE_OFFSET      0x2000u
#define FT_TT_PHASE2_CB_IN0_OFFSET     0x4000u
#define FT_TT_PHASE2_CB_IN1_OFFSET     0x5000u
#define FT_TT_PHASE2_CB_OUT_OFFSET     0x6000u
#define FT_TT_PHASE2_SOURCE0_OFFSET    0xC000u
#define FT_TT_PHASE2_SOURCE1_OFFSET    0xE000u
#define FT_TT_PHASE2_DESTINATION_OFFSET 0x10000u

/** Scratch stride for the runtime-arg probe kernel, one word per TRISC. */
#define FT_TT_PHASE2_PROBE_STRIDE 4u

/** A 32x32 Float16_b tile, the default tile shape the generated CB descriptors describe
 *  (build_tensix_kernel.sh emits Float16_b / 32x32 / 4 faces). */
#define FT_TT_PHASE2_TILE_BYTES 2048u

/** CB indices. These MUST match the compile-time args the kernels were built with —
 *  vector_add is built 0,1 in / 16 out. */
#define FT_TT_PHASE2_CB_IN0 0u
#define FT_TT_PHASE2_CB_IN1 1u
#define FT_TT_PHASE2_CB_OUT 16u

/** Two pages per CB: enough for the FIFO to be a ring rather than a single slot, small
 *  enough that a wrong page count shows up as a wrap bug rather than silence. */
#define FT_TT_PHASE2_CB_PAGE_COUNT 2u
#define FT_TT_PHASE2_CB_BYTES      (FT_TT_PHASE2_CB_PAGE_COUNT * FT_TT_PHASE2_TILE_BYTES)

/** One core's Phase 2 L1 map. Every address is the derived base plus a fixed offset. */
typedef struct ft_tt_phase2_l1_layout {
    uint32_t kernel_config_base; /**< everything the launch message's offsets are relative to */
    uint32_t cb_config_address;
    uint32_t runtime_args_base;
    uint32_t probe_address;
    uint32_t cb_in0_address;
    uint32_t cb_in1_address;
    uint32_t cb_out_address;
    uint32_t source0_address;
    uint32_t source1_address;
    uint32_t destination_address;
    /** No loaded kernel image may reach this address. Same as kernel_config_base; named
     *  separately because that is the question test_phase2_images.c asks. */
    uint32_t allocation_floor;
} ft_tt_phase2_l1_layout;

/**
 * Build the map from where the five FIRMWARE images landed.
 *
 * The base is max(ft_tt_image_placement_l1_floor(placements, count),
 * FT_TT_PHASE2_RESERVED_L1_END), which is the whole point: the first term is what the
 * loader measured on this core with this firmware build, the second is what the memory
 * map reserves whether or not an ELF covers it.
 *
 * @param placements  firmware placements from ft_tt_core_boot_firmware(); entries with
 *                    bytes_written == 0 are ignored. NULL is allowed with count 0, which
 *                    yields the reserved-region floor alone
 * @param count       number of entries
 * @param out_layout  written only on success
 * @return            FT_TT_OK; FT_TT_EINVAL for a NULL out_layout, a floor
 *                    ft_tt_image_placement_l1_floor() refuses, or a base whose highest
 *                    offset would leave L1
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_phase2_l1_layout_derive(const ft_tt_image_placement *placements, unsigned count,
                                  ft_tt_phase2_l1_layout *out_layout);

/**
 * Runtime-arg block address for one processor: the base plus FT_TT_PHASE2_RTA_OFFSET plus
 * `processor` strides. Returns 0 for a NULL layout or an out-of-range processor, which is
 * the "this processor has no args" value the launch encoder already understands.
 */
uint32_t ft_tt_phase2_runtime_args_address(const ft_tt_phase2_l1_layout *layout,
                                           unsigned processor);

#endif /* FT_TT_PHASE2_L1_LAYOUT_H */
