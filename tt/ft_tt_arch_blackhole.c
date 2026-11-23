/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_arch_blackhole.c
 * @brief The Blackhole (p100 / p150) architecture table.
 *
 * Every hardware fact below names where it came from. "Docs" means the public
 * tt-isa-documentation tree (paths relative to its root):
 *   [COORD]  BlackholeA0/NoC/Coordinates.md — coordinate translation table and
 *            the harvested-tile placement rules
 *   [LUA]    Diagrams/Src/NoC.lua — machine-readable tile map: NocSize_Arch.bh,
 *            TileLabel_BH (what sits at each NoC #0 node), and the bh_y1223 /
 *            bh_y25 / bh_x8 translated-numbering examples
 *   [BH]     BlackholeA0/README.md — tile counts per product
 *   [TLB]    BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md
 *   [RESET]  BlackholeA0/TensixTile/SoftReset.md
 *   [RISCV]  BlackholeA0/TensixTile/BabyRISCV/README.md (memory map)
 *   [TILE]   BlackholeA0/TensixTile/README.md
 *   [ETHDMP] BlackholeA0/EthernetTile/Samples/ethdump/ethdump.c (PCI ids)
 * Values no public page states are marked TEST (asserted by a named oracle
 * test) or HEADER (the compiled Foundry header the header gate compares it to,
 * so test_arch's "headers agree" case pins it).
 *
 * @status Complete for every field a Foundry caller reads. tile_height,
 *   tile_width and faces_per_tile are 0: no public page states them and no
 *   caller reads them.
 * @issues None
 * @todo None
 */

#include "ft_tt_arch.h"

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_kmd.h"

/* ------------------------------------------------------------------------- */
/* Identity                                                                  */
/* ------------------------------------------------------------------------- */

/* [ETHDMP] PCI_VENDOR_ID_TENSTORRENT 0x1E52, PCI_DEVICE_ID_BLACKHOLE 0xB140. */
#define BLACKHOLE_PCI_VENDOR_ID 0x1E52u
static const uint16_t blackhole_pci_device_ids[] = { 0xB140u };

/* ------------------------------------------------------------------------- */
/* Tensix                                                                    */
/* ------------------------------------------------------------------------- */

/* [LUA] TileLabel_BH: every node with x not in {0, 8, 9} and y >= 2 is a Tensix
 * ("T"). [COORD] table: Tensix at x 1-7 and 10-16, y 2-11 — 14 x 10 = 140, which
 * [BH] confirms. x = 8 is the ARC/security/CPU column, x = 0 and 9 are DRAM. */
static const uint8_t blackhole_tensix_noc_x[14] = { 1, 2, 3, 4, 5, 6, 7,
                                                    10, 11, 12, 13, 14, 15, 16 };
static const uint8_t blackhole_tensix_noc_y[10] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

/* TEST (tests/test_coord.c harvest_bit_table, tests/test_harvesting.c
 * layout_telemetry_bit_by_column): ARC reports enabled columns outside-in,
 * alternating west then east, so grid column 13 (x = 16) is telemetry bit 1. No
 * public page documents the telemetry encoding; this card reports all 14 bits
 * set, so the order cannot be observed here either. */
static const uint8_t blackhole_tensix_telemetry_bit[14] = { 0, 2, 4, 6, 8, 10, 12,
                                                            13, 11, 9, 7, 5, 3, 1 };

/* ------------------------------------------------------------------------- */
/* DRAM                                                                      */
/* ------------------------------------------------------------------------- */

/* [LUA] TileLabel_BH: DRAM columns x = 0 (D0-D3) and x = 9 (D4-D7); by y the
 * bank is {0,0,1,1,2,3,3,3,2,2,1,0} for y = 0..11, so each bank has 3 tiles
 * ([BH]: 24 DRAM tiles, "each 4 GiB is exposed identically on 3 tiles").
 * Port order within a bank is the order of the translated y each tile receives
 * in the bh_y1223 example (y0->12, y1->13, y11->14, y2->15, y10->16, y3->17,
 * y9->21, y4->22, y8->23, y5->18, y7->19, y6->20), i.e. bank-relative. */
static const ft_tt_arch_xy blackhole_dram_noc0[8 * 3] = {
    { 0, 0 },  { 0, 1 },  { 0, 11 }, /* bank 0 */
    { 0, 2 },  { 0, 10 }, { 0, 3 },  /* bank 1 */
    { 0, 9 },  { 0, 4 },  { 0, 8 },  /* bank 2 */
    { 0, 5 },  { 0, 7 },  { 0, 6 },  /* bank 3 */
    { 9, 0 },  { 9, 1 },  { 9, 11 }, /* bank 4 */
    { 9, 2 },  { 9, 10 }, { 9, 3 },  /* bank 5 */
    { 9, 9 },  { 9, 4 },  { 9, 8 },  /* bank 6 */
    { 9, 5 },  { 9, 7 },  { 9, 6 },  /* bank 7 */
};

/* ------------------------------------------------------------------------- */
/* Ethernet                                                                  */
/* ------------------------------------------------------------------------- */

/* [LUA] TileLabel_BH, y == 1 row: E0 at x 1, E2 x 2, ... E12 x 7, E13 x 10,
 * E11 x 11, ... E1 x 16. Indexed by channel number here. */
static const ft_tt_arch_xy blackhole_eth_noc0[14] = {
    { 1, 1 },  { 16, 1 }, { 2, 1 },  { 15, 1 }, { 3, 1 },  { 14, 1 }, { 4, 1 },
    { 13, 1 }, { 5, 1 },  { 12, 1 }, { 6, 1 },  { 11, 1 }, { 7, 1 },  { 10, 1 },
};

/* ------------------------------------------------------------------------- */
/* Fixed blocks                                                              */
/* ------------------------------------------------------------------------- */

/* [LUA] TileLabel_BH, y == 0 row: "PCIe 0" at x 2, "PCIe 1" at x 11. [BH]: 2 tiles. */
static const ft_tt_arch_xy blackhole_pcie_noc0[2] = { { 2, 0 }, { 11, 0 } };

/* [LUA] TileLabel_BH, x == 8 column: CPU 0-3 at y 3, CPU 8-11 at y 5, CPU 12-15
 * at y 7, CPU 4-7 at y 9. Logical order here is by y. */
static const ft_tt_arch_xy blackhole_l2cpu_noc0[4] = { { 8, 3 }, { 8, 5 }, { 8, 7 }, { 8, 9 } };

/* TEST (tests/test_coord.c test_l2cpu_bit_order): telemetry bit 0 -> (8,3),
 * 1 -> (8,9), 2 -> (8,5), 3 -> (8,7). Consistent with the docs: it is the cluster
 * number implied by the CPU ranges above ([LUA] TileLabel_BH), and the [LUA]
 * bh_x8 example gives exactly those tiles translated y 26, 27, 28, 29. */
static const uint8_t blackhole_l2cpu_telemetry_bit[4] = { 0, 2, 3, 1 };

/* [LUA] TileLabel_BH: the empty ("") labels are NoC nodes with no endpoint tile —
 * twelve in row 0 (x 1-7 and 10-16 minus the two PCIe tiles), (8,1), and
 * (8,4), (8,6), (8,8), (8,10), (8,11) in the CPU column. 18 in all. */
static const ft_tt_arch_xy blackhole_router_noc0[18] = {
    { 1, 0 },  { 3, 0 },  { 4, 0 },  { 5, 0 },  { 6, 0 },  { 7, 0 },
    { 10, 0 }, { 12, 0 }, { 13, 0 }, { 14, 0 }, { 15, 0 }, { 16, 0 },
    { 8, 1 },  { 8, 4 },  { 8, 6 },  { 8, 8 },  { 8, 10 }, { 8, 11 },
};

/* ------------------------------------------------------------------------- */
/* The table                                                                 */
/* ------------------------------------------------------------------------- */

static const ft_tt_arch blackhole_arch = {
    .id = FT_TT_ARCH_BLACKHOLE,
    .name = "Blackhole",
    .pci_vendor_id = BLACKHOLE_PCI_VENDOR_ID,
    .pci_device_ids = blackhole_pci_device_ids,
    .pci_device_id_count = 1u,
    /* HEADER: ft_tt_kmd.h's chip_arch value for Blackhole. */
    .kmd_chip_arch = (uint64_t)FT_TT_KMD_ARCH_BLACKHOLE,

    /* [LUA] NocSize_Arch.bh = { w = 17, h = 12 }. */
    .noc = { .size_x = 17u, .size_y = 12u },

    .tensix = {
        .column_count = 14u,
        .row_count = 10u,
        /* [COORD] "If any Tensix columns are fused off ... X is remapped to put
         * the fused columns at maximal X" — Blackhole harvests columns. */
        .harvest_axis = FT_TT_HARVEST_AXIS_COLUMN,
        .noc_x_by_column = blackhole_tensix_noc_x,
        .noc_y_by_row = blackhole_tensix_noc_y,
        .telemetry_bit_by_column = blackhole_tensix_telemetry_bit,
    },

    .dram = {
        .bank_count = 8u,
        .ports_per_bank = 3u,
        /* [BH] p100 exposes 28 of 32 GiB: at most one 4 GiB bank is fused off. */
        .max_harvested_banks = 1u,
        .noc0 = blackhole_dram_noc0,
        /* [COORD] table: DRAM aliases at x 17-18, y 12-23, 3 y per bank. Which
         * physical column lands on x = 17 for an UNHARVESTED part is TEST
         * (tests/test_coord.c test_dram_full_grid: banks 0-3 at 17). */
        .translated_origin_x = 17u,
        .translated_origin_y = 12u,
        .translated_column_count = 2u,
    },

    .eth = {
        .channel_count = 14u, /* [BH] 14 Ethernet tiles on the full ASIC. */
        .noc0 = blackhole_eth_noc0,
        /* [COORD] table: Ethernet aliases at x 20-31, y 25; [LUA] bh_y25 packs
         * the enabled channels there in channel order. */
        .translated_origin_x = 20u,
        .translated_origin_y = 25u,
    },

    .fixed = {
        .arc = { 8u, 0u },      /* [LUA] TileLabel_BH x 8, y 0 "ARC". */
        .security = { 8u, 2u }, /* [LUA] TileLabel_BH x 8, y 2 "SEC". */
        .security_count = 1u,   /* [BH] */
        .pcie_count = 2u,
        .pcie_noc0 = blackhole_pcie_noc0,
        .pcie_translated = { 19u, 24u }, /* [COORD] "19, 24: PCI Express tile". */
        .l2cpu_count = 4u,               /* [BH] */
        .l2cpu_noc0 = blackhole_l2cpu_noc0,
        .l2cpu_telemetry_bit_by_tile = blackhole_l2cpu_telemetry_bit,
        .router_count = 18u,
        .router_noc0 = blackhole_router_noc0,
    },

    /* [TLB]: 210 windows — 0-201 are 2 MiB (201 reserved for the kernel driver),
     * 202-209 are 4 GiB; config at BAR0 0x1FC0_0000 as 210 x 12-byte entries
     * followed by 32 strided words: 210 * 12 + 32 * 4 = 0xA58 bytes. */
    .tlb = {
        .window_2m_count = 202u,
        .window_4g_count = 8u,
        .kernel_window_index = 201u,
        .register_stride_bytes = 12u,
        .registers_offset = 0x1FC00000u,
        .registers_length = 0xA58u,
        .strided_window_count = 32u,
    },

    /* No public page documents Blackhole ARC telemetry. Each value is what
     * Foundry's card-verified ARC paths use: the reset-unit scratch registers at
     * 0x80030400 with the table pointer in scratch 13 and the values pointer in
     * scratch 12, and the CSM window 0x1000_0000 + 512 KiB (ft_tt_udrv_arc_msg.h,
     * driven on the card by tests/test_udrv_card_arc.c and
     * tests/test_udrv_rehearsal.c; CSM base also in
     * tests/test_phase1b_telemetry.c). Grid tags 34 (Tensix columns),
     * 36 (GDDR), 37 (L2CPU), 40 (NoC translation) are named by tests/test_coord.c;
     * 35 (Ethernet) is the one tag no oracle names — see the report. */
    .telemetry = {
        .scratch_base = 0x80030400u,
        .table_pointer_scratch_index = 13u,
        .values_pointer_scratch_index = 12u,
        .csm_base = 0x10000000u,
        .csm_size = 0x80000u,
        .tag_enabled_tensix_col = 34u,
        .tag_enabled_eth = 35u,
        .tag_enabled_gddr = 36u,
        .tag_enabled_l2cpu = 37u,
        .tag_noc_translation = 40u,
    },

    .core = {
        .processor_count = 5u, /* [TILE] "5x Baby RISCV cores". */
        /* [RISCV] memory map has no NCRISC IRAM row on Blackhole. */
        .has_ncrisc_iram = false,

        /* HEADER (ft_tt_mailbox.h): the firmware-interface contract. */
        .programmable_core_type_count = 4u,
        .max_num_kernels = 3u,
        .subordinate_map_size = 4u,
        .processors_for_threading_variables = 0u,
        .circular_buffer_count = 64u,

        .l1_size = 1536u * 1024u, /* [TILE] "1536 KiB of RAM called L1". */
        .l1_alignment = 16u,      /* HEADER (ft_tt_memmap.h) */
        .dram_alignment = 64u,    /* HEADER (ft_tt_memmap.h) */
        /* HEADER (ft_tt_memmap.h): the usable span of each bank. [BH] gives 4 GiB
         * per bank physically; the compiled map stops 16 MiB short of that. */
        .dram_bytes_per_bank = 0xFF000000u,
        .mailbox_base = 96u, /* HEADER (ft_tt_memmap.h) */
        .soft_reset_address = 0xFFB121B0u, /* [RESET] RISCV_DEBUG_REG_SOFT_RESET_0 */
        /* [RESET] bits 11 (B), 12-14 (T0-T2), 18 (NC). */
        .soft_reset_all_riscs_mask = (1u << 11) | (1u << 12) | (1u << 13) | (1u << 14) |
                                     (1u << 18),
        .noc_overlay_base = 0xFFB40000u,  /* [RISCV] NOC_OVERLAY_START_ADDR */
        .noc_stream_register_span = 0x1000u, /* HEADER (ft_tt_memmap.h) */
        .tile_height = 0u,
        .tile_width = 0u,
        .faces_per_tile = 0u,
    },
};

const ft_tt_arch *ft_tt_arch_blackhole(void)
{
    return &blackhole_arch;
}
