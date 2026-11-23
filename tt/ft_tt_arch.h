/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_arch.h
 * @brief Per-architecture description tables for Tenstorrent parts, selected at
 *   RUNTIME from the PCI vendor/device id the kernel driver reports.
 *
 * One table describes one chip family: NoC grid size, where every tile type sits
 * on NoC #0, how translated coordinates are laid out, how ARC telemetry encodes
 * harvesting, the host TLB windows, and the per-Tensix scalars the compiled
 * memory-map headers were written against.
 *
 * Only Blackhole has a table. Every other Tenstorrent part (Wormhole B0,
 * Grayskull) is recognised BY NAME so a refusal can say what it saw, and is then
 * refused: a table nobody could check against hardware is not shipped. Inside the
 * one table, a field whose value is not verified is 0, never a plausible guess.
 *
 * Nothing here touches a device; every function is a pure function of its
 * arguments and of constant data.
 *
 * @status Blackhole table complete; selection, naming and both self-checks
 *   implemented and covered by tests/test_arch.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ARCH_H
#define FT_TT_ARCH_H

#include <stdbool.h>
#include <stdint.h>

/** Chip families this module can NAME. Naming is not support: only
 *  FT_TT_ARCH_BLACKHOLE ever selects a table. */
typedef enum ft_tt_arch_id {
    FT_TT_ARCH_UNKNOWN = 0,
    FT_TT_ARCH_GRAYSKULL,
    FT_TT_ARCH_WORMHOLE_B0,
    FT_TT_ARCH_BLACKHOLE
} ft_tt_arch_id;

/** Which Tensix axis a part fuses off when it harvests. ft_tt_coord implements
 *  COLUMN only, and ft_tt_arch_check() refuses anything else. */
typedef enum ft_tt_harvest_axis {
    FT_TT_HARVEST_AXIS_NONE = 0,
    FT_TT_HARVEST_AXIS_ROW,
    FT_TT_HARVEST_AXIS_COLUMN
} ft_tt_harvest_axis;

/** One NoC position, compact form for constant tables. */
typedef struct ft_tt_arch_xy {
    uint8_t x;
    uint8_t y;
} ft_tt_arch_xy;

/**
 * The Tensix grid. Grid column i sits at NoC #0 x = noc_x_by_column[i]; grid row j
 * at NoC #0 y = noc_y_by_row[j]. `telemetry_bit_by_column[i]` is the bit of ARC's
 * enabled-columns telemetry word that reports grid column i. It must be a
 * permutation of 0..column_count-1; it is NOT the identity on Blackhole.
 */
typedef struct ft_tt_arch_tensix {
    uint32_t column_count;
    uint32_t row_count;
    ft_tt_harvest_axis harvest_axis;
    const uint8_t *noc_x_by_column;
    const uint8_t *noc_y_by_row;
    const uint8_t *telemetry_bit_by_column;
} ft_tt_arch_tensix;

/** Upper bound on DRAM banks any table may declare. The buffer allocator keeps
 *  one statically sized arena per bank, so a larger part is refused at the table
 *  gate rather than overflowing that array. */
#define FT_TT_MAX_DRAM_BANKS 8u

/**
 * DRAM banks. Each bank is reachable through `ports_per_bank` NoC tiles that all
 * expose the same memory; `noc0` holds bank_count x ports_per_bank entries,
 * bank-major, in port order. In translated space the banks form
 * `translated_column_count` columns starting at (translated_origin_x,
 * translated_origin_y), ports_per_bank consecutive y per bank.
 */
typedef struct ft_tt_arch_dram {
    uint32_t bank_count;
    uint32_t ports_per_bank;
    uint32_t max_harvested_banks;
    const ft_tt_arch_xy *noc0;
    uint32_t translated_origin_x;
    uint32_t translated_origin_y;
    uint32_t translated_column_count;
} ft_tt_arch_dram;

/** Ethernet tiles, indexed by channel. Enabled channels are packed, in channel
 *  order, along one translated row starting at the translated origin. */
typedef struct ft_tt_arch_eth {
    uint32_t channel_count;
    const ft_tt_arch_xy *noc0;
    uint32_t translated_origin_x;
    uint32_t translated_origin_y;
} ft_tt_arch_eth;

/**
 * Tiles that are never harvested-and-renumbered by Foundry: ARC, the security
 * tile, the PCIe endpoints, the L2CPU clusters, and router-only NoC nodes.
 * `l2cpu_noc0` lists L2CPU tiles in Foundry's logical order;
 * `l2cpu_telemetry_bit_by_tile[t]` is the enabled-L2CPU telemetry bit for tile t.
 * `pcie_translated` is the single translated alias of whichever PCIe tile is in
 * use on the board.
 */
typedef struct ft_tt_arch_fixed_blocks {
    ft_tt_arch_xy arc;
    ft_tt_arch_xy security;
    uint32_t security_count;
    uint32_t pcie_count;
    const ft_tt_arch_xy *pcie_noc0;
    ft_tt_arch_xy pcie_translated;
    uint32_t l2cpu_count;
    const ft_tt_arch_xy *l2cpu_noc0;
    const uint8_t *l2cpu_telemetry_bit_by_tile;
    uint32_t router_count;
    const ft_tt_arch_xy *router_noc0;
} ft_tt_arch_fixed_blocks;

/** NoC coordinate space size (nodes, not cores). */
typedef struct ft_tt_arch_noc {
    uint32_t size_x;
    uint32_t size_y;
} ft_tt_arch_noc;

/** Host-to-device TLB windows exposed through the PCIe BARs. */
typedef struct ft_tt_arch_tlb {
    uint32_t window_2m_count;
    uint32_t window_4g_count;
    uint32_t kernel_window_index;
    uint32_t register_stride_bytes;
    uint32_t registers_offset;
    uint32_t registers_length;
    uint32_t strided_window_count;
} ft_tt_arch_tlb;

/** Where ARC publishes its telemetry table, and the tag ids of the five grid
 *  facts Foundry reads from it. */
typedef struct ft_tt_arch_telemetry {
    uint32_t scratch_base;
    uint32_t table_pointer_scratch_index;
    uint32_t values_pointer_scratch_index;
    uint32_t csm_base;
    uint32_t csm_size;
    uint16_t tag_enabled_tensix_col;
    uint16_t tag_enabled_eth;
    uint16_t tag_enabled_gddr;
    uint16_t tag_enabled_l2cpu;
    uint16_t tag_noc_translation;
} ft_tt_arch_telemetry;

/**
 * Per-Tensix scalars. The second group (programmable_core_type_count through
 * circular_buffer_count) is the firmware-interface contract ft_tt_mailbox.h's
 * struct layout is derived from; ft_tt_arch_check_headers() compares those and
 * the memory-map scalars against the compiled headers.
 */
typedef struct ft_tt_arch_tensix_core {
    uint32_t processor_count;
    bool has_ncrisc_iram;

    uint32_t programmable_core_type_count;
    uint32_t max_num_kernels;
    uint32_t subordinate_map_size;
    uint32_t processors_for_threading_variables;
    uint32_t circular_buffer_count;

    uint32_t l1_size;
    uint32_t l1_alignment;
    uint32_t dram_alignment;
    uint64_t dram_bytes_per_bank;
    uint32_t mailbox_base;
    uint32_t soft_reset_address;
    uint32_t soft_reset_all_riscs_mask;
    uint32_t noc_overlay_base;
    uint32_t noc_stream_register_span;
    uint32_t tile_height;
    uint32_t tile_width;
    uint32_t faces_per_tile;
} ft_tt_arch_tensix_core;

/** One architecture. All pointers reference static constant storage. */
typedef struct ft_tt_arch {
    ft_tt_arch_id id;
    const char *name;
    uint16_t pci_vendor_id;
    const uint16_t *pci_device_ids;
    uint32_t pci_device_id_count;
    uint64_t kmd_chip_arch;
    ft_tt_arch_noc noc;
    ft_tt_arch_tensix tensix;
    ft_tt_arch_dram dram;
    ft_tt_arch_eth eth;
    ft_tt_arch_fixed_blocks fixed;
    ft_tt_arch_tlb tlb;
    ft_tt_arch_telemetry telemetry;
    ft_tt_arch_tensix_core core;
} ft_tt_arch;

/** The Blackhole table. Static storage, never NULL, never freed. */
const ft_tt_arch *ft_tt_arch_blackhole(void);

/** Name the family a Tenstorrent PCI device id belongs to, supported or not.
 *  Unrecognised ids are FT_TT_ARCH_UNKNOWN. */
ft_tt_arch_id ft_tt_arch_id_for_pci_device_id(uint16_t pci_device_id);

/** Printable family name. Never NULL, including for out-of-range values. */
const char *ft_tt_arch_id_string(ft_tt_arch_id id);

/**
 * Select the table for a device, cross-checked against the kernel driver's own
 * chip_arch value. Returns NULL (reason logged) for a wrong vendor, an
 * unsupported or unknown device id, a driver opinion that disagrees with the
 * device id, or a table that fails ft_tt_arch_check() / ft_tt_arch_check_headers().
 */
const ft_tt_arch *ft_tt_arch_select(uint16_t pci_vendor_id, uint16_t pci_device_id,
                                    uint64_t kmd_chip_arch);

/** As ft_tt_arch_select() without the driver cross-check, for paths that read the
 *  ids straight from PCI config space with no kernel driver bound. */
const ft_tt_arch *ft_tt_arch_select_by_pci_ids(uint16_t pci_vendor_id, uint16_t pci_device_id);

/**
 * Structural self-check of a table: identity present, harvest axis is COLUMN,
 * bit-order tables are permutations, masks fit 32 bits, every coordinate lies
 * inside the NoC grid, DRAM banks divide into the alias columns and fit
 * FT_TT_MAX_DRAM_BANKS, TLB and reset descriptions non-empty.
 *
 * @return FT_TT_OK, FT_TT_EINVAL (NULL), or FT_TT_EIO (reason logged).
 */
int ft_tt_arch_check(const ft_tt_arch *arch);

/**
 * Check that the compiled single-part headers (ft_tt_memmap.h, ft_tt_mailbox.h,
 * ft_tt_launch.h, ft_tt_device.h) describe the same part as `arch`.
 *
 * @return FT_TT_OK, FT_TT_EINVAL (NULL), or FT_TT_EIO (first mismatch logged).
 */
int ft_tt_arch_check_headers(const ft_tt_arch *arch);

#endif
