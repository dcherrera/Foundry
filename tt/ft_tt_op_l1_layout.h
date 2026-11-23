/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_l1_layout.h
 * @brief Where a Phase 4 op puts things in one Tensix's L1 — kernel text included, which
 *   is the difference from the Phase 2 layout.
 *
 * ---------------------------------------------------------------------------
 * WHY KERNEL TEXT MOVED, AND WHERE UPSTREAM PUTS IT
 * ---------------------------------------------------------------------------
 * Phase 2's layout header says "kernel TEXT is not in this map — it is linked just after
 * its processor's firmware." That is true of `vector_add` and it does not survive contact
 * with a real op. A firmware hole's tail is 1360 B on TRISC0, 1952 B on TRISC1, 1392 B on
 * TRISC2 and 1296 B on NCRISC, and what is above it is the NEXT processor's firmware, not
 * free L1. Measured on the card host 2026-08-12:
 *
 *   vector_add_compute   trisc0 1292  trisc1  780  trisc2 1280   fits, 68 B to spare
 *   matmul_compute       trisc0 1404  trisc1  900  trisc2 1316   TRISC0 over by 44
 *   rms_norm_compute     trisc0 2448  trisc1 2504  trisc2 2548   all three over
 *
 * **Upstream never had this problem, because upstream does not put kernels there.**
 * `ConfigureKernelGroup()` writes every
 * kernel binary at `kernel_config_base + kernel_text_offsets[processor]`, where
 * `kernel_config_base = hal.get_dev_addr(TENSIX, HalL1MemAddrType::KERNEL_CONFIG)`. On
 * Blackhole that is `MEM_MAP_END` with a **69 KiB** region above it
 * (`default_l1_kernel_config_size`), sub-allocated per
 * processor, with the buffer arena (`DEFAULT_UNRESERVED`) starting past its end. So the
 * structure below is theirs: one region, per-processor slabs inside it, CB config and
 * runtime args in the same region, arena after it.
 *
 * **The two deliberate divergences, both stated so nobody "fixes" them:**
 *
 *  1. **Our base is 0x14000, not `MEM_MAP_END` (0x9F80).** `MEM_MAP_END` is also
 *     `MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH`: the per-processor firmware `.data` staging
 *     that `do_crt1()` reads at boot. Upstream overwrites it once firmware is up, because
 *     its cores are booted once by the runtime. Ours are booted by whichever test is
 *     running, and a core re-deasserted after its staging was overwritten boots into
 *     garbage globals with no error anywhere. 0x14000 is above the staging AND the
 *     bank→NoC tables (0x137A0), so a re-boot is always safe. It costs ~41 KiB of L1.
 *  2. **We do not XIP.** `ElfFile::MakeExecuteInPlace()` is one line —
 *     `GetSegments().front().address = 0` — which works because their
 *     text is position-independent and the runtime then places it wherever the program's
 *     allocator chose. We instead LINK each kernel at its slab address
 *     (`--defsym=__fw_export_text_end=<base>`, build_tensix_kernel.sh), so the linker
 *     resolves everything and nothing depends on the text being relocatable. The cost is
 *     that a binary is bound to one slab, which is exactly right while one program at a
 *     time runs on a core, and is what Phase 5 will have to revisit if it wants several
 *     resident programs.
 *
 * A happy consequence of using a real KERNEL_CONFIG region: `kernel_text_offset` is now a
 * small POSITIVE number, the same as upstream's, instead of leaning on the 32-bit unsigned
 * wraparound docs/tt_launch_notes.md §3 describes. The wraparound still works and is still
 * tested; it is simply no longer load-bearing.
 *
 * **THIS HEADER IS THE ONE SOURCE OF THE RELOCATION MAP.** `build_tensix_program.sh`
 * used to carry its own copy of these five addresses with nothing enforcing agreement;
 * since 2026-08-12 it PARSES `FT_TT_OP_KERNEL_CONFIG_BASE` and the five
 * `FT_TT_OP_TEXT_OFFSET_*` out of this file instead, and stops if either is missing.
 * Editing the map here is therefore enough. A disagreement could never corrupt anything
 * — the ELF carries its own load addresses, so the loader writes where the ELF says —
 * but it would leave the launch message pointing at empty L1, which
 * ft_tt_op_image_load()'s slab check turns into a refusal rather than a silent no-op.
 *
 * ---------------------------------------------------------------------------
 * THE MAP
 * ---------------------------------------------------------------------------
 *   0x14000  kernel-config region, 69 KiB, ends 0x25400
 *     +0x0000  CB descriptor array, 64 CBs x 4 words
 *     +0x0400  runtime args, 5 blocks of 0x80
 *     +0x0800  replica table (kernels/resident_output_replicas.h)
 *     +0x1000  BRISC  kernel text   0x15000  16 KiB  \
 *     +0x5000  NCRISC kernel text   0x19000  11 KiB   |  ASYMMETRIC since 2026-11-19, and
 *     +0x7C00  TRISC0 kernel text   0x1BC00  14 KiB   |  filling the region to its end.
 *     +0xB400  TRISC1 kernel text   0x1F400  12 KiB   |  See "WHY THE SLABS ARE NOT
 *     +0xE400  TRISC2 kernel text   0x22400  12 KiB  /   EQUAL" below.
 *   0x26000  circular-buffer arena  (upstream's DEFAULT_UNRESERVED), 1,351,680 B
 *   0x170000 Phase 5 work queue, 64 KiB, to the top of L1 (ft_tt_queue.h)
 *   0x180000 end of L1
 *
 * **WHY THE SLABS ARE NOT EQUAL (2026-11-19).** Until then every slab was 12 KiB and the
 * region's last 5 KiB (0x24000-0x25400) were unused. The images did not grow evenly: BRISC
 * reached 12,276 B (gdn) and its timeline build 13,512 B, TRISC0 12,044 B
 * (attention_k8v4_split), while NCRISC peaked at 8,800 B (topk; 10,008 B with timeline
 * stamps) and TRISC1/2 at ~9 KiB. So BRISC takes the freed tail and 1 KiB from NCRISC, TRISC0
 * takes 2 KiB, and TRISC1/2 keep 12 KiB. Moving a slab
 * relinks every image linked into it, so LOAD-IDENTICAL cannot gate such a change;
 * tools/compare_tensix_relocated_images.py is the gate instead (same program, every
 * differing word explained as the move). ft_tt_op_text_slab_map_valid() is what a map edit
 * must pass: ascending, no overlap, inside the region, past the replica table.
 *
 * **What the arena is worth as a PREFETCH DEPTH**, since that is the only thing in a
 * Phase 4 op whose size is not fixed: the matmul gives each input CB two blocks of
 * `depth` pages and the output CB two pages, so
 * `2*depth*(in0_tile + in1_tile) + 2*out_tile <= 1,351,680` — depth <= **164** for a
 * bf16 weight, **214** for Bfp8_b, **256** for Bfp4_b. MiniCPM5-1B's deepest K is
 * `ffn_down` at Kt = 144, and depth is shrunk to a divisor of Kt, so **this model can
 * never reach the ceiling**. Measured, not only derived: depth 144 on `ffn_down` places
 * and runs, PCC unchanged.
 *
 * Every offset inside the region is under 64 KiB, which the 16-bit `local_cb_offset` and
 * `rta_offset` wire fields require.
 *
 * @status Used by ft_tt_op_matmul.c, ft_tt_op_rms_norm.c, ft_tt_launch.c and tests/test_ops.c.
 *   Asymmetric text slabs since 2026-11-19.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_L1_LAYOUT_H
#define FT_TT_OP_L1_LAYOUT_H

#include <stdint.h>

#include "ft_tt_memmap.h"

/** Everything in the region is an offset from here, including the 16-bit wire fields.
 *  Upstream's equivalent is MEM_MAP_END; divergence 1 in the header note says why ours is
 *  higher. 16-byte aligned (FT_TT_MAX_NOC_WRITE_ALIGNMENT). */
#define FT_TT_OP_KERNEL_CONFIG_BASE 0x00014000u

/** `default_l1_kernel_config_size`. Taken as-is: it is upstream's
 *  own budget for exactly this content, and our five images plus config fit inside it. */
#define FT_TT_OP_KERNEL_CONFIG_BYTES (69u * 1024u)

/** 64 CBs x 4 words, at offset 0 from the config base. */
#define FT_TT_OP_CB_CONFIG_ADDRESS FT_TT_OP_KERNEL_CONFIG_BASE

/** Per-processor runtime-arg blocks. 0x80 apart is 32 args each, more than any Phase 4
 *  kernel takes; every ENABLED processor needs a real block, not just the interesting one
 *  (docs/tt_launch_notes.md §3, "the trap that cost a run"). */
#define FT_TT_OP_RTA_BASE   (FT_TT_OP_KERNEL_CONFIG_BASE + 0x0400u)
#define FT_TT_OP_RTA_STRIDE 0x80u
#define FT_TT_OP_RTA_ADDRESS(processor) \
    (FT_TT_OP_RTA_BASE + (uint32_t)(processor) * FT_TT_OP_RTA_STRIDE)

/** Per-processor kernel text slabs, as offsets from the config base, and their sizes. Mirrors
 *  the map build_tensix_program.sh applies under RELOCATE_KERNEL_TEXT=1 (it parses the
 *  offsets out of this file). A slab holds the image's text AND its .data load image. */
#define FT_TT_OP_TEXT_OFFSET_DM0   0x1000u
#define FT_TT_OP_TEXT_OFFSET_DM1   0x5000u
#define FT_TT_OP_TEXT_OFFSET_MATH0 0x7C00u
#define FT_TT_OP_TEXT_OFFSET_MATH1 0xB400u
#define FT_TT_OP_TEXT_OFFSET_MATH2 0xE400u
#define FT_TT_OP_TEXT_SLAB_BYTES_DM0   0x4000u
#define FT_TT_OP_TEXT_SLAB_BYTES_DM1   0x2C00u
#define FT_TT_OP_TEXT_SLAB_BYTES_MATH0 0x3800u
#define FT_TT_OP_TEXT_SLAB_BYTES_MATH1 0x3000u
#define FT_TT_OP_TEXT_SLAB_BYTES_MATH2 0x3000u

/** The first config-region offset a text slab may use: past the CB array, the runtime-arg
 *  blocks and the replica table (FT_TT_REPLICA_TABLE_ADDRESS, asserted in ft_tt_op_replicas.c). */
#define FT_TT_OP_TEXT_REGION_FIRST_OFFSET 0x1000u

/** Slab alignment: the NoC write alignment, and what the loader's segment writes assume. */
#define FT_TT_OP_TEXT_SLAB_ALIGN 0x10u

/** Where circular-buffer FIFOs start, and the first address past the arena. The base is
 *  the region end rounded up, i.e. upstream's DEFAULT_UNRESERVED rule.
 *
 *  The arena ENDS at the Phase 5 work queue rather than at the top of L1. Putting the
 *  queue at the top and growing CBs up from the bottom is what keeps the two from
 *  needing to know about each other: an op's CB placement is unchanged, and the only
 *  coupling is this one subtraction. A resident kernel that also uses CBs (Phase 6) gets
 *  the same arena every Phase 4 op had, minus 64 KiB. */
#define FT_TT_OP_CB_ARENA_BASE 0x00026000u
#define FT_TT_OP_CB_ARENA_END  FT_TT_OP_QUEUE_BASE

/** The Phase 5 work-queue region: control block plus descriptor ring, at the TOP of L1.
 *  64 KiB holds a 1024-slot ring of 32-byte descriptors with room over. Its internal
 *  layout is ft_tt_queue.h's business, not this map's. */
#define FT_TT_OP_QUEUE_BYTES 0x00010000u
#define FT_TT_OP_QUEUE_BASE  (FT_TT_MEM_L1_SIZE - FT_TT_OP_QUEUE_BYTES) /* 0x170000 */

/** A 32x32 tile, per format. The block-float sizes are mantissa bytes plus a 64-byte
 *  shared-exponent section, and they are what foundry/convert/ft_bfp.c
 *  emits — so an artifact tile is a byte copy into a CB, with no conversion at either
 *  end. That is Route B's payoff (spec §6.3) stated as a number. */
#define FT_TT_OP_TILE_BYTES_FP32   4096u
#define FT_TT_OP_TILE_BYTES_BF16   2048u
#define FT_TT_OP_TILE_BYTES_BFP8_B 1088u
#define FT_TT_OP_TILE_BYTES_BFP4_B 576u

/** Elements down and across one tile. */
#define FT_TT_OP_TILE_DIM 32u

/**
 * Whether a text-slab map is legal: every slab aligned, non-empty, ascending in processor
 * order, not overlapping the next, and wholly inside [FIRST_OFFSET, region size). Takes the
 * map as arrays so a host test can feed it a broken one; the real map is checked at compile
 * time below.
 *
 * @param offsets  FT_TT_PROC_COUNT slab offsets from FT_TT_OP_KERNEL_CONFIG_BASE
 * @param bytes    FT_TT_PROC_COUNT slab sizes
 * @return         1 legal, 0 not
 */
static inline int ft_tt_op_text_slab_map_valid(const uint32_t *offsets, const uint32_t *bytes)
{
    unsigned processor;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t end = offsets[processor] + bytes[processor];

        if (bytes[processor] == 0u || offsets[processor] % FT_TT_OP_TEXT_SLAB_ALIGN != 0u ||
            bytes[processor] % FT_TT_OP_TEXT_SLAB_ALIGN != 0u)
            return 0;
        if (offsets[processor] < FT_TT_OP_TEXT_REGION_FIRST_OFFSET ||
            end > FT_TT_OP_KERNEL_CONFIG_BYTES || end < offsets[processor])
            return 0;
        if (processor + 1u < FT_TT_PROC_COUNT && end > offsets[processor + 1u])
            return 0;
    }
    return 1;
}

/* The shipped map, checked where the compiler can refuse it (the same rules as above). */
FT_TT_STATIC_ASSERT(FT_TT_OP_TEXT_OFFSET_DM0 >= FT_TT_OP_TEXT_REGION_FIRST_OFFSET,
                    text_slab_dm0_past_the_reserved_head);
FT_TT_STATIC_ASSERT(FT_TT_OP_TEXT_OFFSET_DM0 + FT_TT_OP_TEXT_SLAB_BYTES_DM0 <=
                        FT_TT_OP_TEXT_OFFSET_DM1, text_slab_dm0_clears_dm1);
FT_TT_STATIC_ASSERT(FT_TT_OP_TEXT_OFFSET_DM1 + FT_TT_OP_TEXT_SLAB_BYTES_DM1 <=
                        FT_TT_OP_TEXT_OFFSET_MATH0, text_slab_dm1_clears_math0);
FT_TT_STATIC_ASSERT(FT_TT_OP_TEXT_OFFSET_MATH0 + FT_TT_OP_TEXT_SLAB_BYTES_MATH0 <=
                        FT_TT_OP_TEXT_OFFSET_MATH1, text_slab_math0_clears_math1);
FT_TT_STATIC_ASSERT(FT_TT_OP_TEXT_OFFSET_MATH1 + FT_TT_OP_TEXT_SLAB_BYTES_MATH1 <=
                        FT_TT_OP_TEXT_OFFSET_MATH2, text_slab_math1_clears_math2);
FT_TT_STATIC_ASSERT(FT_TT_OP_TEXT_OFFSET_MATH2 + FT_TT_OP_TEXT_SLAB_BYTES_MATH2 <=
                        FT_TT_OP_KERNEL_CONFIG_BYTES, text_slab_math2_inside_the_region);

/** The kernel text slab size for an FT_TT_PROC_* index, or 0 if the index is out of range. */
static inline uint32_t ft_tt_op_text_slab_bytes(unsigned processor)
{
    switch (processor) {
    case FT_TT_PROC_DM0:
        return FT_TT_OP_TEXT_SLAB_BYTES_DM0;
    case FT_TT_PROC_DM1:
        return FT_TT_OP_TEXT_SLAB_BYTES_DM1;
    case FT_TT_PROC_MATH0:
        return FT_TT_OP_TEXT_SLAB_BYTES_MATH0;
    case FT_TT_PROC_MATH1:
        return FT_TT_OP_TEXT_SLAB_BYTES_MATH1;
    case FT_TT_PROC_MATH2:
        return FT_TT_OP_TEXT_SLAB_BYTES_MATH2;
    default:
        return 0u;
    }
}

/** The kernel text base for an FT_TT_PROC_* index, or 0 if the index is out of range. */
static inline uint32_t ft_tt_op_text_base(unsigned processor)
{
    switch (processor) {
    case FT_TT_PROC_DM0:
        return FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_TEXT_OFFSET_DM0;
    case FT_TT_PROC_DM1:
        return FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_TEXT_OFFSET_DM1;
    case FT_TT_PROC_MATH0:
        return FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_TEXT_OFFSET_MATH0;
    case FT_TT_PROC_MATH1:
        return FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_TEXT_OFFSET_MATH1;
    case FT_TT_PROC_MATH2:
        return FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_TEXT_OFFSET_MATH2;
    default:
        return 0u;
    }
}

#endif /* FT_TT_OP_L1_LAYOUT_H */
