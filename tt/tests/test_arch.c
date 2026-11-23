/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_arch.c
 * @brief Off-hardware test for the architecture seam (ft_tt_arch.{h,c},
 *   ft_tt_arch_blackhole.c): what gets selected, what gets REFUSED, and whether
 *   a refusal says enough to act on.
 *
 * Most of this file tests failure. That is the point. The seam exists because
 * we have one Tenstorrent part and can verify values for no other, so the
 * behaviour that actually protects a stranger's card is the refusal — and a
 * refusal nobody has watched happen is worth as little as a green test nobody
 * has watched fail (coding-best-practices.md §9).
 *
 * TOUCHES NO DEVICE. `ft_tt_arch_select()` is a pure function of three
 * integers, which is what lets every rejection path be exercised on a machine
 * with no card in it — including the Wormhole and Grayskull ones we could never
 * reach otherwise.
 *
 * @status Written 2026-08-12. Every case has been observed RED — see
 *   docs/tt_driver_notes.md §8.8.
 * @issues None
 * @todo None
 *
 * Sources for the ids asserted below (captured 2026-08-12):
 *   SRC16 = the kernel driver's device enumeration — PCI_VENDOR_ID_TENSTORRENT and
 *           the Grayskull / Wormhole / Blackhole device ids; the userspace
 *           device-id -> arch switch is the same mapping, which this mirrors
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_kmd.h"

/* SRC16:15-18. The vendor id and the three device ids, transliterated here
 * independently of ft_tt_arch_blackhole.c so the comparison means something. */
#define TENSTORRENT_VENDOR_ID 0x1E52u
#define GRAYSKULL_DEVICE_ID   0xFACAu
#define WORMHOLE_DEVICE_ID    0x401Eu
#define BLACKHOLE_DEVICE_ID   0xB140u

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u32(uint32_t got, uint32_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = %u\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s = %u (expected %u)\n", what, got, want);
    }
}

/* ------------------------------------------------------------------------- */
/* Selection                                                                 */
/* ------------------------------------------------------------------------- */

static void test_blackhole_is_selected(void)
{
    const ft_tt_arch *arch;

    printf("selecting the one part we have\n");
    arch = ft_tt_arch_select(TENSTORRENT_VENDOR_ID, BLACKHOLE_DEVICE_ID,
                             (uint64_t)FT_TT_KMD_ARCH_BLACKHOLE);
    check(arch != NULL, "a Blackhole at 0x1E52:0xB140 selects a table");
    check(arch == ft_tt_arch_blackhole(), "and it is the Blackhole table");
    check(arch != NULL && arch->id == FT_TT_ARCH_BLACKHOLE, "with the Blackhole id");
    check(arch != NULL && strcmp(arch->name, "Blackhole") == 0, "and a name a log can print");
}

/*
 * Every one of these must return NULL. None of them can be reached on this
 * host, which is exactly why they are worth writing down: the first time a
 * Wormhole meets this code should not also be the first time this branch runs.
 */
static void test_every_other_part_is_refused(void)
{
    printf("refusals (stderr above each is the message a user would get)\n");

    check(ft_tt_arch_select(TENSTORRENT_VENDOR_ID, WORMHOLE_DEVICE_ID,
                            (uint64_t)FT_TT_KMD_ARCH_WORMHOLE) == NULL,
          "a Wormhole B0 is refused — we have no verified tables for it");
    check(ft_tt_arch_select(TENSTORRENT_VENDOR_ID, GRAYSKULL_DEVICE_ID,
                            (uint64_t)FT_TT_KMD_ARCH_UNKNOWN) == NULL,
          "a Grayskull is refused");
    check(ft_tt_arch_select(TENSTORRENT_VENDOR_ID, 0x0000u, (uint64_t)FT_TT_KMD_ARCH_BLACKHOLE) ==
              NULL,
          "an unrecognised Tenstorrent device id is refused");
    check(ft_tt_arch_select(0x10DEu, BLACKHOLE_DEVICE_ID, (uint64_t)FT_TT_KMD_ARCH_BLACKHOLE) ==
              NULL,
          "the right device id under the WRONG vendor is refused");

    /* The driver's opinion is a cross-check, not a second lookup. If the two
     * disagree, one is wrong and we cannot tell which — which is not a state to
     * aim a TLB window from (spec §7.3). */
    check(ft_tt_arch_select(TENSTORRENT_VENDOR_ID, BLACKHOLE_DEVICE_ID,
                            (uint64_t)FT_TT_KMD_ARCH_WORMHOLE) == NULL,
          "a Blackhole device id with a Wormhole chip_arch is refused as a disagreement");
    check(ft_tt_arch_select(TENSTORRENT_VENDOR_ID, BLACKHOLE_DEVICE_ID,
                            (uint64_t)FT_TT_KMD_ARCH_UNKNOWN) == NULL,
          "a Blackhole device id with an UNKNOWN chip_arch is refused too");
}

/*
 * A refusal that cannot name the part is a hex number and a shrug. These assert
 * the naming layer separately from the selection layer, because the naming is
 * the whole reason the unsupported ids are in the tree at all.
 */
static void test_a_refusal_can_name_what_it_saw(void)
{
    printf("naming an unsupported part\n");
    check(ft_tt_arch_id_for_pci_device_id(WORMHOLE_DEVICE_ID) == FT_TT_ARCH_WORMHOLE_B0,
          "0x401E is named as Wormhole B0");
    check(ft_tt_arch_id_for_pci_device_id(GRAYSKULL_DEVICE_ID) == FT_TT_ARCH_GRAYSKULL,
          "0xFACA is named as Grayskull");
    check(ft_tt_arch_id_for_pci_device_id(BLACKHOLE_DEVICE_ID) == FT_TT_ARCH_BLACKHOLE,
          "0xB140 is named as Blackhole");
    check(ft_tt_arch_id_for_pci_device_id(0x0000u) == FT_TT_ARCH_UNKNOWN,
          "an id we cannot name says so rather than guessing");

    check(strcmp(ft_tt_arch_id_string(FT_TT_ARCH_WORMHOLE_B0), "Wormhole B0") == 0,
          "Wormhole B0 has a printable name");
    check(strcmp(ft_tt_arch_id_string(FT_TT_ARCH_GRAYSKULL), "Grayskull") == 0,
          "Grayskull has a printable name");
    check(strcmp(ft_tt_arch_id_string(FT_TT_ARCH_UNKNOWN), "unknown") == 0,
          "an unknown part has a printable name");
    check(ft_tt_arch_id_string((ft_tt_arch_id)99) != NULL,
          "an out-of-range id still returns a string, never NULL");
}

/* ------------------------------------------------------------------------- */
/* The table gate                                                            */
/* ------------------------------------------------------------------------- */

/*
 * ft_tt_arch_check() is what a future port has to satisfy. Feeding it damaged
 * copies of the real table is the only way to know it would catch a
 * half-filled one — a check run exclusively against a correct table has never
 * demonstrated anything.
 */
static void test_the_table_gate_catches_a_broken_table(void)
{
    const ft_tt_arch *good = ft_tt_arch_blackhole();
    ft_tt_arch broken;

    printf("the arch table self-check\n");
    check(ft_tt_arch_check(good) == FT_TT_OK, "the Blackhole table passes");
    check(ft_tt_arch_check(NULL) == FT_TT_EINVAL, "NULL is refused");

    broken = *good;
    broken.tensix.telemetry_bit_by_column = NULL;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "a missing harvesting bit order is refused, NOT read as the identity");

    broken = *good;
    broken.tensix.harvest_axis = FT_TT_HARVEST_AXIS_ROW;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "a ROW-harvesting part is refused — ft_tt_coord implements columns only");

    broken = *good;
    broken.tensix.harvest_axis = FT_TT_HARVEST_AXIS_NONE;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "an unset harvest axis is refused");

    broken = *good;
    broken.noc.size_x = 8u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "a NoC grid too small for the table's own coordinates is refused");

    broken = *good;
    broken.dram.translated_column_count = 3u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "8 DRAM banks that do not divide into the alias columns is refused");

    broken = *good;
    broken.dram.noc0 = NULL;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "a DRAM bank count with no table is refused");

    broken = *good;
    broken.fixed.l2cpu_telemetry_bit_by_tile = NULL;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "L2CPU tiles with no telemetry bit order is refused");

    broken = *good;
    broken.tlb.window_2m_count = 0u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "an empty TLB description is refused");

    broken = *good;
    broken.core.soft_reset_all_riscs_mask = 0u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "a per-Tensix description with no reset mask is refused");

    broken = *good;
    broken.pci_device_id_count = 0u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "a table with no identity is refused");

    /* The telemetry masks are uint32_t and the accessors build them with
     * `(1u << count) - 1`, which is undefined at 32. A wider part needs a wider
     * mask type, not a bigger constant — so the gate refuses rather than
     * silently shifting off the end. */
    broken = *good;
    broken.tensix.column_count = 32u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "32 Tensix columns is refused (32-bit mask)");

    broken = *good;
    broken.dram.bank_count = 32u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "32 DRAM banks is refused (32-bit mask)");

    broken = *good;
    broken.eth.channel_count = 32u;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO, "32 ETH channels is refused (32-bit mask)");

    /* The allocator's per-bank arenas are static, caller-owned storage sized by
     * FT_TT_MAX_DRAM_BANKS. A part with more banks than that must be refused HERE,
     * at device open, rather than overflow the array on the first allocation. The
     * bound is under the 31-bank mask ceiling above, so this is a distinct gate and
     * needs its own case. */
    broken = *good;
    broken.dram.bank_count = FT_TT_MAX_DRAM_BANKS + 1u;
    broken.dram.translated_column_count = 1u; /* keep the divisibility rule satisfied */
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "more DRAM banks than the allocator can hold is refused at the table gate");

    check(FT_TT_MAX_DRAM_BANKS >= good->dram.bank_count,
          "the capacity is at least what the one implemented part declares");
}

/*
 * The harvesting table must be a PERMUTATION. A duplicate entry harvests one
 * column twice and leaves another unreachable, and every coordinate it hands
 * out still looks like a real Tensix.
 */
static void test_the_gate_catches_a_bad_permutation(void)
{
    const ft_tt_arch *good = ft_tt_arch_blackhole();
    uint8_t duplicated[14] = { 0, 2, 4, 6, 8, 10, 12, 13, 11, 9, 7, 5, 3, 1 };
    uint8_t out_of_range[14] = { 0, 2, 4, 6, 8, 10, 12, 13, 11, 9, 7, 5, 3, 1 };
    ft_tt_arch broken;

    printf("the harvesting table must be a permutation\n");

    duplicated[13] = 0u; /* column 13 now claims column 0's bit */
    broken = *good;
    broken.tensix.telemetry_bit_by_column = duplicated;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "two columns claiming one telemetry bit is refused");

    out_of_range[7] = 14u; /* outside the 14-bit mask */
    broken = *good;
    broken.tensix.telemetry_bit_by_column = out_of_range;
    check(ft_tt_arch_check(&broken) == FT_TT_EIO,
          "a harvesting bit outside the mask width is refused");

    {
        uint8_t l2cpu_bits[4] = { 0, 2, 3, 32 };

        broken = *good;
        broken.fixed.l2cpu_telemetry_bit_by_tile = l2cpu_bits;
        check(ft_tt_arch_check(&broken) == FT_TT_EIO,
              "an L2CPU telemetry bit past 31 is refused — same shifting hazard");
    }
}

/* ------------------------------------------------------------------------- */
/* The header gate                                                           */
/* ------------------------------------------------------------------------- */

/*
 * ft_tt_memmap.h, ft_tt_mailbox.h and ft_tt_launch.h are transliterations of
 * ONE part's headers and cannot describe two at once. This gate is what stops
 * the seam from selecting an architecture and then applying another one's L1
 * map — the failure mode that would corrupt a core rather than report an error.
 *
 * The Blackhole comparison is not a self-comparison: ft_tt_arch_blackhole.c
 * transliterates these scalars a second time, independently of the headers, so
 * a bad edit to either side goes red.
 */
static void test_the_header_gate(void)
{
    const ft_tt_arch *good = ft_tt_arch_blackhole();
    ft_tt_arch broken;

    printf("the compiled headers must describe the selected part\n");
    check(ft_tt_arch_check_headers(good) == FT_TT_OK,
          "the compiled headers agree with the Blackhole table");
    check(ft_tt_arch_check_headers(NULL) == FT_TT_EINVAL, "NULL is refused");

    broken = *good;
    broken.core.l1_size = 1024u * 1024u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different L1 size is caught — the memory map would be wrong");

    broken = *good;
    broken.core.processor_count = 4u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different processor count is caught — the mailbox padding depends on it");

    broken = *good;
    broken.core.mailbox_base = 0u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO, "a different mailbox base is caught");

    /*
     * The other five core_config.h constants ft_tt_mailbox.h's STRUCT LAYOUT is
     * derived from. All five are identical on Wormhole today, so nothing would
     * exercise them by accident — which is the reason they need a test at all.
     * A disagreement does not shift one field, it moves every offset behind it,
     * and the symptom is the firmware reading endpoint tables out of someone
     * else's bytes on a core that then needs a host reboot.
     */
    broken = *good;
    broken.core.programmable_core_type_count = 5u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different programmable-core-type count is caught — it sizes kernel_config_base[]");

    broken = *good;
    broken.core.max_num_kernels = 4u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different kernel count is caught — it sizes shared_globals_ready[]");

    broken = *good;
    broken.core.subordinate_map_size = 8u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different subordinate map size is caught — it moves every field after it");

    broken = *good;
    broken.core.processors_for_threading_variables = 5u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a part WITH threading variables is caught — the two arrays are omitted here");

    broken = *good;
    broken.core.circular_buffer_count = 32u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "Wormhole's 32 circular buffers is caught against Blackhole's 64");

    broken = *good;
    broken.core.soft_reset_address = 0xFFB12000u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different soft-reset address is caught");

    broken = *good;
    broken.core.soft_reset_all_riscs_mask = 0x1Fu;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different all-RISC reset mask is caught — the bits are non-contiguous");

    broken = *good;
    broken.tensix.row_count = 12u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a different row count is caught against FT_TT_TENSIX_ROWS_PER_COLUMN");

    broken = *good;
    broken.fixed.arc.x = 9u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "ARC x disagreeing between tt-kmd and tt-umd is caught — the one live-verified pair");

    broken = *good;
    broken.fixed.arc.y = 1u;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO, "ARC y disagreeing is caught");

    /* NCRISC IRAM is a presence flag, not a number: the loader has no staging
     * step for it, so no amount of matching constants makes a part with IRAM
     * bootable by this code. */
    broken = *good;
    broken.core.has_ncrisc_iram = true;
    check(ft_tt_arch_check_headers(&broken) == FT_TT_EIO,
          "a part with NCRISC IRAM is refused — the loader has no path for it");
}

/* ------------------------------------------------------------------------- */
/* The seam actually carries the geometry                                    */
/* ------------------------------------------------------------------------- */

/*
 * The seam is only real if the coordinate layer READS it. Handing ft_tt_coord a
 * grid whose architecture has a deliberately shortened column list must change
 * the answers — if it does not, the tables were moved but the constants were
 * not, and the whole exercise was decoration.
 */
static void test_coord_reads_the_table_not_a_constant(void)
{
    static const uint8_t four_columns[4] = { 1, 2, 3, 4 };
    static const uint8_t four_bits[4] = { 0, 1, 2, 3 };
    static const uint8_t three_rows[3] = { 2, 3, 4 };
    const ft_tt_arch *blackhole = ft_tt_arch_blackhole();
    ft_tt_arch small = *blackhole;
    ft_tt_grid_info grid;
    ft_tt_noc_coord c;

    small.tensix.column_count = 4u;
    small.tensix.row_count = 3u;
    small.tensix.noc_x_by_column = four_columns;
    small.tensix.noc_y_by_row = three_rows;
    small.tensix.telemetry_bit_by_column = four_bits;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    grid.arch = &small;
    grid.tensix_enabled_col = 0xFu;
    grid.enabled_gddr = 0xFFu;
    grid.enabled_eth = 0x3FFFu;
    grid.enabled_l2cpu = 0xFu;
    grid.noc_translation_enabled = 1u;

    printf("ft_tt_coord answers from the table, not from a compiled-in 14x10\n");
    check(ft_tt_arch_check(&small) == FT_TT_OK, "the 4x3 table is structurally valid");
    check_u32(ft_tt_coord_tensix_column_count(&grid), 4u, "columns follow the table");
    check_u32(ft_tt_coord_tensix_row_count(&grid), 3u, "rows follow the table");
    check_u32(ft_tt_coord_tensix_core_count(&grid), 12u, "core count is 4 x 3, not 140");
    check(ft_tt_coord_tensix_at(&grid, 0, 3, FT_TT_COORD_NOC0, &c) == FT_TT_EINVAL,
          "row 3 is refused on a 3-row table");
    check(ft_tt_coord_tensix_at(&grid, 4, 0, FT_TT_COORD_NOC0, &c) == FT_TT_EINVAL,
          "column 4 is refused on a 4-column table");

    /* And the header gate refuses the same table, because ft_tt_memmap.h still
     * says 10 rows. A geometry the coordinate layer can express is not
     * automatically one the compiled kernels can run on. */
    check(ft_tt_arch_check_headers(&small) == FT_TT_EIO,
          "but the header gate refuses it — the memory map still says 10 rows");
}

/*
 * A grid with no architecture is a set of telemetry words with no meaning. It
 * must refuse, not fall back to whichever part happened to be compiled in —
 * that fallback is precisely how a Blackhole coordinate would reach a Wormhole.
 */
static void test_a_grid_without_an_architecture_is_refused(void)
{
    ft_tt_grid_info orphan;
    ft_tt_noc_coord c;

    memset(&orphan, 0, sizeof(orphan));
    orphan.valid = true;
    orphan.tensix_enabled_col = 0x3FFFu;
    orphan.enabled_gddr = 0xFFu;

    printf("a grid with no architecture\n");
    check(ft_tt_coord_check_grid(&orphan) == FT_TT_EUNAVAIL,
          "check_grid refuses it rather than assuming Blackhole");
    check_u32(ft_tt_coord_tensix_column_count(&orphan), 0u, "no column count is produced");
    check_u32(ft_tt_coord_tensix_core_count(&orphan), 0u, "no core count is produced");
    check(ft_tt_coord_tensix_at(&orphan, 0, 0, FT_TT_COORD_NOC0, &c) == FT_TT_EUNAVAIL,
          "no Tensix coordinate is produced");
    check(ft_tt_coord_dram_bank(&orphan, 0, FT_TT_COORD_NOC0, &c) == FT_TT_EUNAVAIL,
          "no DRAM coordinate is produced");
    check(ft_tt_coord_classify(&orphan, FT_TT_COORD_NOC0, 1, 2) == FT_TT_CORE_NONE,
          "classify answers NONE rather than guessing a block type");
    check(!ft_tt_coord_is_enabled(&orphan, FT_TT_COORD_NOC0, 1, 2), "is_enabled says no");
    check(ft_tt_coord_arc(NULL, FT_TT_COORD_NOC0, &c) == FT_TT_EINVAL,
          "ARC refuses a NULL architecture");
    check(ft_tt_coord_pcie(NULL, 0, FT_TT_COORD_NOC0, &c) == FT_TT_EINVAL,
          "PCIe refuses a NULL architecture");
    {
        uint32_t mask = 0;

        check(ft_tt_coord_tensix_harvest_mask(NULL, 0x3FFFu, &mask) == FT_TT_EINVAL,
              "the harvest mask refuses a NULL architecture instead of answering 0");
    }
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("ft_tt_arch seam test — no device is touched\n");
    printf("Most of this is refusals; they are the behaviour that protects a card.\n\n");

    test_blackhole_is_selected();
    test_every_other_part_is_refused();
    test_a_refusal_can_name_what_it_saw();
    test_the_table_gate_catches_a_broken_table();
    test_the_gate_catches_a_bad_permutation();
    test_the_header_gate();
    test_coord_reads_the_table_not_a_constant();
    test_a_grid_without_an_architecture_is_refused();

    printf("\n%d/%d checks failed\n", checks_failed, checks_run);
    return checks_failed == 0 ? 0 : 1;
}
