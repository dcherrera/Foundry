/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_arch.c
 * @brief Architecture selection by PCI id, family naming, and the two table
 *   self-checks (structural, and agreement with the compiled headers).
 *
 * Selection is deliberately narrow: exactly one family has a table, and any other
 * Tenstorrent device id is named and refused. The table the selector hands out
 * has already passed both checks, so a caller holding a non-NULL table can index
 * its arrays by its own counts without re-validating.
 *
 * @status Complete; covered by tests/test_arch.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_arch.h"

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_device.h"
#include "ft_tt_launch.h"
#include "ft_tt_mailbox.h"
#include "ft_tt_memmap.h"

/** Tenstorrent's PCI vendor id (public docs: ethdump.c sample). */
#define TENSTORRENT_PCI_VENDOR_ID 0x1E52u

/** Device ids of the families we can name. Blackhole is in the docs (ethdump.c
 *  sample); Wormhole B0 and Grayskull are TEST (tests/test_arch.c). */
#define BLACKHOLE_PCI_DEVICE_ID 0xB140u
#define WORMHOLE_PCI_DEVICE_ID  0x401Eu
#define GRAYSKULL_PCI_DEVICE_ID 0xFACAu

/** Telemetry masks are 32-bit words built as (1u << count) - 1, so every
 *  per-bit count must stay below 32. */
#define MAX_TELEMETRY_MASK_BITS 31u

/* ------------------------------------------------------------------------- */
/* Naming                                                                    */
/* ------------------------------------------------------------------------- */

ft_tt_arch_id ft_tt_arch_id_for_pci_device_id(uint16_t pci_device_id)
{
    switch (pci_device_id) {
    case BLACKHOLE_PCI_DEVICE_ID:
        return FT_TT_ARCH_BLACKHOLE;
    case WORMHOLE_PCI_DEVICE_ID:
        return FT_TT_ARCH_WORMHOLE_B0;
    case GRAYSKULL_PCI_DEVICE_ID:
        return FT_TT_ARCH_GRAYSKULL;
    default:
        return FT_TT_ARCH_UNKNOWN;
    }
}

const char *ft_tt_arch_id_string(ft_tt_arch_id id)
{
    switch (id) {
    case FT_TT_ARCH_GRAYSKULL:
        return "Grayskull";
    case FT_TT_ARCH_WORMHOLE_B0:
        return "Wormhole B0";
    case FT_TT_ARCH_BLACKHOLE:
        return "Blackhole";
    case FT_TT_ARCH_UNKNOWN:
    default:
        return "unknown";
    }
}

/* ------------------------------------------------------------------------- */
/* Selection                                                                 */
/* ------------------------------------------------------------------------- */

/** The supported table for a family, or NULL when the family has none. */
static const ft_tt_arch *supported_table_for_family(ft_tt_arch_id id)
{
    if (id == FT_TT_ARCH_BLACKHOLE)
        return ft_tt_arch_blackhole();
    return NULL;
}

const ft_tt_arch *ft_tt_arch_select_by_pci_ids(uint16_t pci_vendor_id, uint16_t pci_device_id)
{
    ft_tt_arch_id family;
    const ft_tt_arch *table;

    if (pci_vendor_id != TENSTORRENT_PCI_VENDOR_ID) {
        ft_tt_log_error("unsupported architecture: PCI vendor 0x%04x is not Tenstorrent "
                        "(device 0x%04x)",
                        (unsigned)pci_vendor_id, (unsigned)pci_device_id);
        return NULL;
    }

    family = ft_tt_arch_id_for_pci_device_id(pci_device_id);
    table = supported_table_for_family(family);
    if (!table) {
        ft_tt_log_error("unsupported architecture: %s (PCI 0x%04x:0x%04x) — no verified "
                        "table exists for it",
                        ft_tt_arch_id_string(family), (unsigned)pci_vendor_id,
                        (unsigned)pci_device_id);
        return NULL;
    }

    if (ft_tt_arch_check(table) != FT_TT_OK || ft_tt_arch_check_headers(table) != FT_TT_OK) {
        ft_tt_log_error("unsupported architecture: %s table failed its self-check",
                        table->name);
        return NULL;
    }
    return table;
}

const ft_tt_arch *ft_tt_arch_select(uint16_t pci_vendor_id, uint16_t pci_device_id,
                                    uint64_t kmd_chip_arch)
{
    const ft_tt_arch *table = ft_tt_arch_select_by_pci_ids(pci_vendor_id, pci_device_id);

    if (!table)
        return NULL;

    /* The driver's view is a cross-check. Two sources that disagree mean one of
     * them is wrong, and we cannot tell which. */
    if (table->kmd_chip_arch != kmd_chip_arch) {
        ft_tt_log_error("unsupported architecture: device id says %s but the kernel driver "
                        "reports chip_arch %llu (expected %llu)",
                        table->name, (unsigned long long)kmd_chip_arch,
                        (unsigned long long)table->kmd_chip_arch);
        return NULL;
    }
    return table;
}

/* ------------------------------------------------------------------------- */
/* Structural check                                                          */
/* ------------------------------------------------------------------------- */

/** True when `bits[0..count)` is a permutation of 0..count-1. */
static bool is_bit_permutation(const uint8_t *bits, uint32_t count)
{
    uint32_t seen = 0u;
    uint32_t index;

    for (index = 0; index < count; index++) {
        if (bits[index] >= count || (seen & (1u << bits[index])))
            return false;
        seen |= 1u << bits[index];
    }
    return true;
}

/** True when every entry of a coordinate list lies inside the NoC grid. */
static bool xy_list_inside_grid(const ft_tt_arch *arch, const ft_tt_arch_xy *list, uint32_t count)
{
    uint32_t index;

    for (index = 0; index < count; index++) {
        if (list[index].x >= arch->noc.size_x || list[index].y >= arch->noc.size_y)
            return false;
    }
    return true;
}

static bool xy_inside_grid(const ft_tt_arch *arch, ft_tt_arch_xy point)
{
    return point.x < arch->noc.size_x && point.y < arch->noc.size_y;
}

static int refuse_table(const ft_tt_arch *arch, const char *reason)
{
    ft_tt_log_error("arch table '%s' refused: %s", arch->name ? arch->name : "(unnamed)", reason);
    return FT_TT_EIO;
}

static int check_identity(const ft_tt_arch *arch)
{
    if (!arch->name || arch->pci_vendor_id == 0u || !arch->pci_device_ids ||
        arch->pci_device_id_count == 0u)
        return refuse_table(arch, "no identity (name / PCI ids)");
    if (arch->noc.size_x == 0u || arch->noc.size_y == 0u)
        return refuse_table(arch, "empty NoC grid");
    return FT_TT_OK;
}

static int check_tensix(const ft_tt_arch *arch)
{
    const ft_tt_arch_tensix *tensix = &arch->tensix;
    uint32_t index;

    if (tensix->harvest_axis != FT_TT_HARVEST_AXIS_COLUMN)
        return refuse_table(arch, "harvest axis is not COLUMN (the only one ft_tt_coord maps)");
    if (tensix->column_count == 0u || tensix->row_count == 0u)
        return refuse_table(arch, "empty Tensix grid");
    if (tensix->column_count > MAX_TELEMETRY_MASK_BITS)
        return refuse_table(arch, "Tensix columns do not fit a 32-bit telemetry mask");
    if (!tensix->noc_x_by_column || !tensix->noc_y_by_row || !tensix->telemetry_bit_by_column)
        return refuse_table(arch, "missing Tensix column / row / harvesting-bit table");
    if (!is_bit_permutation(tensix->telemetry_bit_by_column, tensix->column_count))
        return refuse_table(arch, "Tensix harvesting bit order is not a permutation");

    for (index = 0; index < tensix->column_count; index++) {
        if (tensix->noc_x_by_column[index] >= arch->noc.size_x)
            return refuse_table(arch, "a Tensix column lies outside the NoC grid");
    }
    for (index = 0; index < tensix->row_count; index++) {
        if (tensix->noc_y_by_row[index] >= arch->noc.size_y)
            return refuse_table(arch, "a Tensix row lies outside the NoC grid");
    }
    return FT_TT_OK;
}

static int check_dram(const ft_tt_arch *arch)
{
    const ft_tt_arch_dram *dram = &arch->dram;

    if (dram->bank_count > MAX_TELEMETRY_MASK_BITS)
        return refuse_table(arch, "DRAM banks do not fit a 32-bit telemetry mask");
    if (dram->bank_count > FT_TT_MAX_DRAM_BANKS)
        return refuse_table(arch, "more DRAM banks than FT_TT_MAX_DRAM_BANKS");
    if (dram->bank_count == 0u)
        return FT_TT_OK;
    if (!dram->noc0 || dram->ports_per_bank == 0u)
        return refuse_table(arch, "DRAM banks declared with no port table");
    if (dram->translated_column_count == 0u ||
        dram->bank_count % dram->translated_column_count != 0u)
        return refuse_table(arch, "DRAM banks do not divide into the translated alias columns");
    if (dram->max_harvested_banks >= dram->bank_count)
        return refuse_table(arch, "every DRAM bank may be harvested");
    if (!xy_list_inside_grid(arch, dram->noc0, dram->bank_count * dram->ports_per_bank))
        return refuse_table(arch, "a DRAM port lies outside the NoC grid");
    return FT_TT_OK;
}

static int check_eth(const ft_tt_arch *arch)
{
    if (arch->eth.channel_count > MAX_TELEMETRY_MASK_BITS)
        return refuse_table(arch, "ETH channels do not fit a 32-bit telemetry mask");
    if (arch->eth.channel_count == 0u)
        return FT_TT_OK;
    if (!arch->eth.noc0)
        return refuse_table(arch, "ETH channels declared with no table");
    if (!xy_list_inside_grid(arch, arch->eth.noc0, arch->eth.channel_count))
        return refuse_table(arch, "an ETH channel lies outside the NoC grid");
    return FT_TT_OK;
}

static int check_fixed_blocks(const ft_tt_arch *arch)
{
    const ft_tt_arch_fixed_blocks *fixed = &arch->fixed;

    if (!xy_inside_grid(arch, fixed->arc))
        return refuse_table(arch, "ARC lies outside the NoC grid");
    if (fixed->security_count > 1u ||
        (fixed->security_count == 1u && !xy_inside_grid(arch, fixed->security)))
        return refuse_table(arch, "bad security tile description");
    if (fixed->pcie_count > 0u &&
        (!fixed->pcie_noc0 || !xy_list_inside_grid(arch, fixed->pcie_noc0, fixed->pcie_count)))
        return refuse_table(arch, "bad PCIe tile table");
    if (fixed->router_count > 0u &&
        (!fixed->router_noc0 ||
         !xy_list_inside_grid(arch, fixed->router_noc0, fixed->router_count)))
        return refuse_table(arch, "bad router node table");
    if (fixed->l2cpu_count == 0u)
        return FT_TT_OK;
    if (fixed->l2cpu_count > MAX_TELEMETRY_MASK_BITS)
        return refuse_table(arch, "L2CPU tiles do not fit a 32-bit telemetry mask");
    if (!fixed->l2cpu_noc0 || !fixed->l2cpu_telemetry_bit_by_tile)
        return refuse_table(arch, "L2CPU tiles declared with no table / telemetry bit order");
    if (!is_bit_permutation(fixed->l2cpu_telemetry_bit_by_tile, fixed->l2cpu_count))
        return refuse_table(arch, "L2CPU telemetry bit order is not a permutation");
    if (!xy_list_inside_grid(arch, fixed->l2cpu_noc0, fixed->l2cpu_count))
        return refuse_table(arch, "an L2CPU tile lies outside the NoC grid");
    return FT_TT_OK;
}

static int check_tlb_and_core(const ft_tt_arch *arch)
{
    if (arch->tlb.window_2m_count == 0u || arch->tlb.register_stride_bytes == 0u ||
        arch->tlb.kernel_window_index >= arch->tlb.window_2m_count ||
        arch->tlb.strided_window_count > arch->tlb.window_2m_count)
        return refuse_table(arch, "empty or inconsistent TLB description");
    if (arch->core.processor_count == 0u || arch->core.l1_size == 0u ||
        arch->core.soft_reset_all_riscs_mask == 0u)
        return refuse_table(arch, "per-Tensix description missing processors / L1 / reset mask");
    return FT_TT_OK;
}

int ft_tt_arch_check(const ft_tt_arch *arch)
{
    int rc;

    if (!arch)
        return FT_TT_EINVAL;

    rc = check_identity(arch);
    if (rc == FT_TT_OK)
        rc = check_tensix(arch);
    if (rc == FT_TT_OK)
        rc = check_dram(arch);
    if (rc == FT_TT_OK)
        rc = check_eth(arch);
    if (rc == FT_TT_OK)
        rc = check_fixed_blocks(arch);
    if (rc == FT_TT_OK)
        rc = check_tlb_and_core(arch);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Header agreement                                                          */
/* ------------------------------------------------------------------------- */

/** One scalar the compiled headers assume, paired with the table's value. */
typedef struct header_expectation {
    const char *what;
    uint64_t table_value;
    uint64_t header_value;
} header_expectation;

int ft_tt_arch_check_headers(const ft_tt_arch *arch)
{
    size_t index;

    if (!arch)
        return FT_TT_EINVAL;

    {
        const ft_tt_arch_tensix_core *core = &arch->core;
        const header_expectation expectations[] = {
            { "L1 size", core->l1_size, FT_TT_MEM_L1_SIZE },
            { "processor count", core->processor_count, FT_TT_PROC_COUNT },
            { "processors per core type", core->processor_count,
              FT_TT_MAX_PROCESSORS_PER_CORE_TYPE },
            { "mailbox base", core->mailbox_base, FT_TT_MEM_MAILBOX_BASE },
            { "programmable core types", core->programmable_core_type_count,
              FT_TT_PROGRAMMABLE_CORE_TYPE_COUNT },
            { "max kernels", core->max_num_kernels, FT_TT_MAX_NUM_KERNELS },
            { "subordinate map size", core->subordinate_map_size, FT_TT_SUBORDINATE_MAP_SIZE },
            { "threading-variable processors", core->processors_for_threading_variables,
              FT_TT_MAX_PROCESSORS_FOR_THREADING_VARIABLES },
            { "circular buffers", core->circular_buffer_count, FT_TT_NUM_CIRCULAR_BUFFERS },
            { "soft-reset address", core->soft_reset_address, FT_TT_TENSIX_SOFT_RESET_ADDR },
            { "all-RISC reset mask", core->soft_reset_all_riscs_mask, FT_TT_SOFT_RESET_ALL_RISCS },
            { "DRAM bytes per bank", core->dram_bytes_per_bank, FT_TT_MEM_DRAM_SIZE },
            { "L1 alignment", core->l1_alignment, FT_TT_L1_ALIGNMENT },
            { "DRAM alignment", core->dram_alignment, FT_TT_DRAM_ALIGNMENT },
            { "NoC overlay base", core->noc_overlay_base, FT_TT_NOC_OVERLAY_BASE },
            { "NoC stream register span", core->noc_stream_register_span,
              FT_TT_NOC_STREAM_REG_SPACE_SIZE },
            { "Tensix rows", arch->tensix.row_count, FT_TT_TENSIX_ROWS_PER_COLUMN },
            { "harvest axis", (uint64_t)arch->tensix.harvest_axis, FT_TT_TENSIX_HARVEST_AXIS },
            { "NoC size x", arch->noc.size_x, FT_TT_NOC_SIZE_X },
            { "NoC size y", arch->noc.size_y, FT_TT_NOC_SIZE_Y },
            { "ARC x", arch->fixed.arc.x, FT_TT_ARC_NOC0_X },
            { "ARC y", arch->fixed.arc.y, FT_TT_ARC_NOC0_Y },
            { "2 MiB TLB windows", arch->tlb.window_2m_count, FT_TT_TLB_2M_WINDOW_COUNT },
            { "4 GiB TLB windows", arch->tlb.window_4g_count, FT_TT_TLB_4G_WINDOW_COUNT },
            { "kernel TLB window", arch->tlb.kernel_window_index, FT_TT_TLB_KERNEL_WINDOW_INDEX },
            { "TLB register stride", arch->tlb.register_stride_bytes, FT_TT_TLB_REG_STRIDE },
            { "TLB register offset", arch->tlb.registers_offset, FT_TT_TLB_REGS_OFFSET },
            { "strided TLB windows", arch->tlb.strided_window_count,
              FT_TT_TLB_STRIDED_WINDOW_COUNT },
        };

        for (index = 0; index < sizeof(expectations) / sizeof(expectations[0]); index++) {
            if (expectations[index].table_value != expectations[index].header_value) {
                ft_tt_log_error("arch table '%s' disagrees with the compiled headers: %s is "
                                "%llu in the table, %llu in the headers",
                                arch->name ? arch->name : "(unnamed)", expectations[index].what,
                                (unsigned long long)expectations[index].table_value,
                                (unsigned long long)expectations[index].header_value);
                return FT_TT_EIO;
            }
        }
    }

    /* The register file the headers map must cover every TLB register. */
    if (arch->tlb.registers_length > FT_TT_TLB_REGS_LEN)
        return refuse_table(arch, "TLB registers extend past the mapped register window");
    /* A presence flag, not a number: the loader has no path for NCRISC IRAM. */
    if (arch->core.has_ncrisc_iram)
        return refuse_table(arch, "part has NCRISC IRAM, which the loader cannot stage");
    return FT_TT_OK;
}
