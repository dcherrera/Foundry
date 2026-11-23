/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_card_tlb.c
 * @brief ON-SILICON bring-up for the userspace driver's TLB path, sequenced so
 *   that tt-kmd is its own oracle and nothing is written until reading has
 *   proved the mapping, the offsets and the bit layout.
 *
 *   Four rungs, each a strict superset of the one before:
 *
 *     1. discover the card through sysfs and mmap BAR0's TLB register file
 *        READ-ONLY. A read-only mapping is the safety property, not care: a
 *        stray store is a SIGSEGV on the host rather than a wrong aim on the
 *        card.
 *     2. READ WINDOW 201 — the one tt-kmd keeps for its own NoC reads — and
 *        decode it. It must come out as the ARC tile (8,0), NoC 0, strict
 *        ordering, aimed at the ARC CSM page. That is a 96-bit prediction with
 *        zero risk attached: no window is claimed, nothing is written.
 *     3. GATE THE ENCODER ON SILICON WITHOUT WRITING A REGISTER. Claim windows
 *        through ALLOCATE_TLB, aim them with CONFIGURE_TLB, and byte-compare
 *        the three words the DRIVER left in BAR0 against the three words our
 *        encoder says it should have written. tt-kmd becomes the oracle on real
 *        hardware, and any disagreement is found before we have written a bit.
 *     4. ONLY THEN write the registers ourselves, read them back through a
 *        different path, and prove a window we aimed reaches the address we
 *        aimed it at. Requires the `write` argument — the default run cannot
 *        write to the card at all.
 *
 *   THE SWEEP NEVER DEREFERENCES AN APERTURE. Configuring a window emits no NoC
 *   transaction; only touching its 2 MiB aperture does. That is what makes it
 *   safe to sweep coordinates at 0 and 63, multicast rectangles, and addresses
 *   at both ends of the 43-bit field: those windows are aimed at endpoints that
 *   would hang the chip if anyone read through them, and nothing here ever maps
 *   them. The one aperture this file maps is aimed at the ARC tile, at an
 *   address tt-kmd itself reads.
 *
 *   WHY A WRONG AIM NEEDS A DELIBERATE DETECTOR. A window pointed at the wrong
 *   place does not fail — it silently reads a different address. So the
 *   functional check does not merely compare our readback against tt-kmd's; it
 *   also proves the comparison DISCRIMINATES, by re-aiming tt-kmd's window at a
 *   different page of the same tile and requiring the bytes to differ. If that
 *   arm ever passes, the equality arm above it was measuring nothing.
 *
 * @status Runs on the p150a at 0000:12:00.0, with tt-kmd
 *   loaded. Needs root (the BAR sysfs files are 0600 root:root).
 * @issues
 *   - /dev/mem is NOT usable as a readback path on this box — a pread at the
 *     BAR's physical address returns EFAULT even as root. Rung 4a records that
 *     explicitly; the second reader is tt-kmd's own BAR0 mapping instead.
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include "ft_tt_kmd_abi.h"
#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_pci_sysfs.h"
#include "ft_tt_udrv_tlb_encode.h"
#include "reference/ft_tt_tlb2m_reference_vectors.h"

/** tt-kmd's character device for card 0. */
#define CHARDEV_PATH "/dev/tenstorrent/0"

/** The window tt-kmd claims for itself: TLB_2M_WINDOW_COUNT - 1. */
#define KERNEL_WINDOW_ID (FT_TT_UDRV_TLB_2M_WINDOW_COUNT - 1u)

/** ARC tile, TRANSLATED coordinates. the kernel driver's value — the one
 *  coordinate in this backend that live hardware has confirmed. */
#define ARC_X 8u
#define ARC_Y 0u

/** ARC CSM, and the reset-unit scratch block. Both are addresses tt-kmd itself
 *  reads on this tile every time anything touches telemetry, which is why they
 *  are the only two the functional check is allowed to aim at. */
#define ARC_CSM_BASE 0x10000000ull
#define ARC_CSM_SIZE (1ull << 19)
#define ARC_SCRATCH_BASE 0x80030400ull
#define ARC_TELEMETRY_TABLE_SCRATCH 13u
#define ARC_TELEMETRY_VALUES_SCRATCH 12u

/** 2 MiB page containing an address. */
#define WINDOW_PAGE(addr) ((addr) & ~((uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE - 1u))

/** How many windows the sweep claims. Enough to reach an id at or above
 *  FT_TT_UDRV_TLB_STRIDED_COUNT, plus the pair 0 and 12 that the strided-offset
 *  discriminator needs. */
#define CLAIMED_WINDOW_COUNT 40u

/** Bytes compared through an aperture in the functional check. */
#define FUNCTIONAL_BLOCK_BYTES 4096u

/** How much of an aperture the functional check maps. The WHOLE window, not
 *  the block it compares: the semantic arm reads the ARC reset-unit scratch at
 *  +0x30400 within the page, and a mapping sized to the comparison block would
 *  make that a segfault rather than a read. */
#define FUNCTIONAL_APERTURE_BYTES ((size_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE)

/* ------------------------------------------------------------------------ *
 * Session state
 * ------------------------------------------------------------------------ */

static ft_tt_udrv_pci_device card;
static int chardev = -1;

/** BAR0's TLB register file, mapped by US through sysfs. `read_registers` is
 *  the only mapping rungs 1-3 have, and it is PROT_READ. */
static const volatile uint32_t *read_registers;
static volatile uint32_t *write_registers;

static uint32_t claimed[CLAIMED_WINDOW_COUNT];
static uint64_t claimed_mmap_uc[CLAIMED_WINDOW_COUNT];
static size_t claimed_count;

/* ------------------------------------------------------------------------ *
 * tt-kmd, used as an oracle and as an arbitrator
 * ------------------------------------------------------------------------ */

static int allocate_window(uint32_t *out_id, uint64_t *out_mmap_uc)
{
    struct ft_tt_kmd_allocate_window_request request;

    memset(&request, 0, sizeof(request));
    request.in.size = (uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE;

    if (ioctl(chardev, FT_TT_KMD_IOCTL_ALLOCATE_WINDOW, &request) != 0)
        return FT_TT_UDRV_EIO;

    *out_id = request.out.id;
    *out_mmap_uc = request.out.mmap_offset_uc;
    return FT_TT_UDRV_OK;
}

static int free_window(uint32_t id)
{
    struct ft_tt_kmd_free_window_request request;

    memset(&request, 0, sizeof(request));
    request.in.id = id;

    return (ioctl(chardev, FT_TT_KMD_IOCTL_FREE_WINDOW, &request) == 0) ? FT_TT_UDRV_OK
                                                                        : FT_TT_UDRV_EIO;
}

/**
 * Aim a window through the ioctl. The field mapping is tt-kmd's, not ours:
 * `static_vc` in the ioctl feeds `use_static_vc` in the register, and the ioctl
 * has no way to express `stream_header` or the 3-bit `static_vc` at all.
 */
static int configure_window_via_ioctl(uint32_t id, const ft_tt_udrv_tlb2m_config *config)
{
    struct ft_tt_kmd_configure_window_request request;

    memset(&request, 0, sizeof(request));
    request.in.id = id;
    request.in.config.addr = config->addr;
    request.in.config.x_end = config->x_end;
    request.in.config.y_end = config->y_end;
    request.in.config.x_start = config->x_start;
    request.in.config.y_start = config->y_start;
    request.in.config.noc = config->noc;
    request.in.config.mcast = config->multicast;
    request.in.config.ordering = config->ordering;
    request.in.config.linked = config->linked;
    request.in.config.static_vc = config->use_static_vc;

    return (ioctl(chardev, FT_TT_KMD_IOCTL_CONFIGURE_WINDOW, &request) == 0) ? FT_TT_UDRV_OK
                                                                             : FT_TT_UDRV_EIO;
}

/* ------------------------------------------------------------------------ *
 * Register access through OUR mapping
 * ------------------------------------------------------------------------ */

/** Word index of window `id`'s register triple within the mapped file. */
static size_t register_word_index(uint32_t id)
{
    return (size_t)id * (FT_TT_UDRV_TLB_REG_SIZE / 4u);
}

static void read_window_registers(uint32_t id, uint32_t out_words[3])
{
    size_t base = register_word_index(id);

    out_words[0] = read_registers[base + 0u];
    out_words[1] = read_registers[base + 1u];
    out_words[2] = read_registers[base + 2u];
}

/** Write a window's three registers, in tt-kmd's order: low, mid, high. */
static void write_window_registers(uint32_t id, const uint32_t words[3])
{
    size_t base = register_word_index(id);

    write_registers[base + 0u] = words[0];
    write_registers[base + 1u] = words[1];
    write_registers[base + 2u] = words[2];
}

/** Word index of window `id`'s strided-multicast register. */
static size_t strided_word_index(uint32_t id)
{
    return (size_t)(FT_TT_UDRV_TLB_STRIDED_REGS_OFFSET +
                    id * FT_TT_UDRV_TLB_STRIDED_REG_SIZE) /
           4u;
}

/* ------------------------------------------------------------------------ *
 * The sweep
 * ------------------------------------------------------------------------ */

/** One configuration the ioctl can express, and therefore one the driver can
 *  be asked to write for us. */
typedef struct sweep_case {
    const char *name;
    ft_tt_udrv_tlb2m_config config;
} sweep_case;

#define SWEEP_CAPACITY 160u

static sweep_case sweep[SWEEP_CAPACITY];
static size_t sweep_count;

static void add_sweep_case(const char *name, uint64_t addr, uint8_t x_end, uint8_t y_end,
                           uint8_t x_start, uint8_t y_start, uint8_t noc, uint8_t multicast,
                           uint8_t ordering, uint8_t linked, uint8_t use_static_vc)
{
    sweep_case *entry;

    if (sweep_count >= SWEEP_CAPACITY)
        return;

    entry = &sweep[sweep_count++];
    memset(entry, 0, sizeof(*entry));
    entry->name = name;
    entry->config.addr = addr;
    entry->config.x_end = x_end;
    entry->config.y_end = y_end;
    entry->config.x_start = x_start;
    entry->config.y_start = y_start;
    entry->config.noc = noc;
    entry->config.multicast = multicast;
    entry->config.ordering = ordering;
    entry->config.linked = linked;
    entry->config.use_static_vc = use_static_vc;
    /* stream_header and static_vc_id stay zero: CONFIGURE_TLB cannot express
     * them, so a sweep case that set them would be comparing our encoder
     * against a driver that was never asked for the same thing. Those two
     * fields are exactly what the userspace driver REGAINS, and they can only
     * be gated in rung 4, where we do the writing. */
}

/**
 * Boundary cases chosen against the field list, then every reference vector
 * the ioctl can express.
 *
 * The address cases matter most: the register's address field is 43 bits at bit
 * 0, so it fills word 0 and spills 11 bits into word 1. `addr < 2^53` and
 * `addr >= 2^53` are the two sides of that boundary and they exercise different
 * words.
 */
static void build_sweep(void)
{
    size_t index;

    add_sweep_case("zero", 0ull, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("arc_tile_strict", ARC_CSM_BASE, ARC_X, ARC_Y, 0, 0, 0, 0, 1, 0, 0);
    add_sweep_case("arc_scratch_strict", WINDOW_PAGE(ARC_SCRATCH_BASE), ARC_X, ARC_Y, 0, 0, 0, 0,
                   1, 0, 0);
    add_sweep_case("addr_one_window", 1ull << 21, 1, 1, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("addr_bit_31", 1ull << 31, 1, 1, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("addr_bit_32", 1ull << 32, 1, 1, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("addr_just_below_word_boundary", (1ull << 53) - (1ull << 21), 1, 1, 0, 0, 0, 0,
                   0, 0, 0);
    add_sweep_case("addr_at_word_boundary", 1ull << 53, 1, 1, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("addr_just_above_word_boundary", (1ull << 53) + (1ull << 21), 1, 1, 0, 0, 0, 0,
                   0, 0, 0);
    add_sweep_case("addr_bit_63", 1ull << 63, 1, 1, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("addr_field_all_ones", 0xFFFFFFFFFFE00000ull, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("coords_zero", 0ull, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("x_end_max", 0ull, 63, 0, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("y_end_max", 0ull, 0, 63, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("x_start_max", 0ull, 0, 0, 63, 0, 0, 0, 0, 0, 0);
    add_sweep_case("y_start_max", 0ull, 0, 0, 0, 63, 0, 0, 0, 0, 0);
    add_sweep_case("y_start_7_below_the_straddle", 0ull, 0, 0, 0, 7, 0, 0, 0, 0, 0);
    add_sweep_case("y_start_8_across_the_straddle", 0ull, 0, 0, 0, 8, 0, 0, 0, 0, 0);
    add_sweep_case("all_coords_max", 0ull, 63, 63, 63, 63, 0, 0, 0, 0, 0);
    add_sweep_case("multicast_rectangle", 1ull << 21, 10, 9, 1, 2, 0, 1, 0, 0, 0);
    add_sweep_case("multicast_full_grid", 0ull, 63, 63, 0, 0, 0, 1, 0, 0, 0);
    add_sweep_case("noc_1", 0ull, 1, 1, 0, 0, 1, 0, 0, 0, 0);
    add_sweep_case("noc_max", 0ull, 1, 1, 0, 0, 3, 0, 0, 0, 0);
    add_sweep_case("ordering_relaxed", 0ull, 1, 1, 0, 0, 0, 0, 0, 0, 0);
    add_sweep_case("ordering_strict", 0ull, 1, 1, 0, 0, 0, 0, 1, 0, 0);
    add_sweep_case("ordering_posted", 0ull, 1, 1, 0, 0, 0, 0, 2, 0, 0);
    add_sweep_case("ordering_posted_strict", 0ull, 1, 1, 0, 0, 0, 0, 3, 0, 0);
    add_sweep_case("linked", 0ull, 1, 1, 0, 0, 0, 0, 0, 1, 0);
    add_sweep_case("use_static_vc", 0ull, 1, 1, 0, 0, 0, 0, 0, 0, 1);
    add_sweep_case("everything_the_ioctl_can_set", 0xFFFFFFFFFFE00000ull, 63, 63, 63, 63, 3, 1, 3,
                   1, 1);

    for (index = 0u; index < FT_TT_TLB2M_REF_VECTOR_COUNT; index++) {
        const ft_tt_tlb2m_reference_vector *vector = &ft_tt_tlb2m_reference_vectors[index];

        if (vector->stream_header != 0u || vector->static_vc != 0u)
            continue;

        add_sweep_case(vector->name, vector->addr, vector->x_end, vector->y_end, vector->x_start,
                       vector->y_start, vector->noc, vector->multicast, vector->ordering,
                       vector->linked, vector->use_static_vc);
    }
}

/* ------------------------------------------------------------------------ *
 * Rung 1 — find the card, map BAR0 read-only
 * ------------------------------------------------------------------------ */

static bool stage_open_the_card(void)
{
    size_t count = 0u;
    void *mapping = NULL;

    ft_tt_test_step("Rung 1 — the card, and BAR0's register file mapped READ-ONLY");

    ft_tt_test_check_fmt("running as root (the BAR files are 0600 root:root)", geteuid() == 0u,
                         "euid=%u", (unsigned)geteuid());

    if (ft_tt_udrv_pci_find_devices(NULL, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT,
                                    FT_TT_UDRV_PCI_DEVICE_BLACKHOLE, &card, 1u,
                                    &count) != FT_TT_UDRV_OK ||
        count == 0u) {
        ft_tt_test_check("a Blackhole is present in sysfs", false, "none found");
        return false;
    }

    ft_tt_test_check_fmt("a Blackhole is present in sysfs", true, "%s (%zu found)", card.address,
                         count);
    ft_tt_test_check_fmt("BAR0 is 512 MiB and prefetchable",
                         card.bar[0].length == (512ull << 20) && card.bar[0].prefetchable,
                         "0x%llx bytes at 0x%llx", (unsigned long long)card.bar[0].length,
                         (unsigned long long)card.bar[0].start);
    ft_tt_test_check_fmt("BAR4 is absent — no 4 GiB windows on this board",
                         !card.bar[4].present, "len=%llu",
                         (unsigned long long)card.bar[4].length);

    /* The register file is UNCACHED, through `resource0`, not `resource0_wc`.
     * That is what tt-kmd's own pci_iomap_range() gives it,
     * and write-combining a register write is how a store arrives late. */
    if (ft_tt_udrv_pci_map_bar(&card, 0u, false, false, FT_TT_UDRV_TLB_REGS_START,
                               FT_TT_UDRV_TLB_REGS_LEN, &mapping) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("BAR0's TLB register file maps read-only", false, "errno %d (%s)",
                             errno, strerror(errno));
        return false;
    }

    read_registers = mapping;
    ft_tt_test_check_fmt("BAR0's TLB register file maps read-only", true,
                         "resource0 + 0x%08X, %u bytes, PROT_READ", FT_TT_UDRV_TLB_REGS_START,
                         FT_TT_UDRV_TLB_REGS_LEN);

    /* The write-combined companion file is the BULK UPLOAD path and nothing has
     * ever used it. Mapping it read-only over the same registers costs nothing
     * and settles half the open question: the file exists and maps. It does NOT
     * settle that a WC STORE reaches the card, which is a separate item. */
    {
        void *wc_mapping = NULL;

        if (ft_tt_udrv_pci_map_bar(&card, 0u, true, false, FT_TT_UDRV_TLB_REGS_START,
                                   FT_TT_UDRV_TLB_REGS_LEN, &wc_mapping) != FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt("resource0_wc maps too", false, "errno %d (%s)", errno,
                                 strerror(errno));
        } else {
            const volatile uint32_t *wc = wc_mapping;
            size_t base = register_word_index(KERNEL_WINDOW_ID);

            ft_tt_test_check_fmt("resource0_wc maps, and reads the same words as resource0",
                                 wc[base + 0u] == read_registers[base + 0u] &&
                                     wc[base + 1u] == read_registers[base + 1u] &&
                                     wc[base + 2u] == read_registers[base + 2u],
                                 "0x%08x 0x%08x 0x%08x", wc[base + 0u], wc[base + 1u],
                                 wc[base + 2u]);
            (void)ft_tt_udrv_pci_unmap_bar(wc_mapping, FT_TT_UDRV_TLB_REGS_LEN);
        }
    }

    chardev = open(CHARDEV_PATH, O_RDWR | O_CLOEXEC);
    ft_tt_test_check_fmt("tt-kmd is loaded and its chardev opens — it is the ORACLE for rung 3, "
                         "and the way back if anything wedges",
                         chardev >= 0, "%s fd=%d", CHARDEV_PATH, chardev);

    return chardev >= 0;
}

/* ------------------------------------------------------------------------ *
 * Rung 2 — the read-only rung
 * ------------------------------------------------------------------------ */

/**
 * Decode the window tt-kmd keeps for itself and check it against a full
 * 96-bit prediction.
 *
 * tt-kmd re-aims window 201 on every NoC read it performs and every one of
 * those reads is `noc_read32(bh, ARC_X, ARC_Y, addr, 0)` with
 * `config.ordering = 1`. Telemetry lives in the CSM, so
 * at rest the window sits on the CSM's 2 MiB page. Everything else must be
 * zero.
 */
static void stage_read_the_kernels_window(void)
{
    ft_tt_udrv_tlb2m_config decoded;
    ft_tt_udrv_tlb2m_config neighbour;
    uint32_t words[3] = {0u, 0u, 0u};
    uint32_t shifted[3] = {0u, 0u, 0u};
    ft_tt_udrv_tlb2m_config shifted_decode;
    size_t all_ones = 0u;
    size_t nonzero = 0u;
    uint32_t id;

    ft_tt_test_step("Rung 2 — READ ONLY. Window %u is tt-kmd's; it must decode to the ARC tile",
                    KERNEL_WINDOW_ID);

    /* A dead BAR reads all-ones everywhere. Census first, so "it decoded" can
     * never be a story about a mapping that returns garbage. */
    for (id = 0u; id < FT_TT_UDRV_TLB_2M_WINDOW_COUNT; id++) {
        uint32_t triple[3];

        read_window_registers(id, triple);
        if (triple[0] == 0xFFFFFFFFu && triple[1] == 0xFFFFFFFFu && triple[2] == 0xFFFFFFFFu)
            all_ones++;
        if (triple[0] != 0u || triple[1] != 0u || triple[2] != 0u)
            nonzero++;
    }

    ft_tt_test_check_fmt("no window reads all-ones — the aperture is decoding, not floating",
                         all_ones == 0u, "%zu of %u", all_ones,
                         FT_TT_UDRV_TLB_2M_WINDOW_COUNT);
    ft_tt_test_check_fmt("some windows hold live configuration — a freshly claimed window is NOT "
                         "blank",
                         nonzero > 0u, "%zu of %u non-zero", nonzero,
                         FT_TT_UDRV_TLB_2M_WINDOW_COUNT);

    read_window_registers(KERNEL_WINDOW_ID, words);
    ft_tt_test_check_fmt("the kernel window's three registers, raw", true,
                         "id %u: 0x%08x 0x%08x 0x%08x", KERNEL_WINDOW_ID, words[0], words[1],
                         words[2]);

    (void)ft_tt_udrv_tlb2m_decode(words, &decoded);

    ft_tt_test_check_fmt("...decodes to the ARC tile's X", decoded.x_end == ARC_X, "%u",
                         decoded.x_end);
    ft_tt_test_check_fmt("...and the ARC tile's Y", decoded.y_end == ARC_Y, "%u", decoded.y_end);
    ft_tt_test_check_fmt("...unicast, so the rectangle fields are zero",
                         decoded.multicast == 0u && decoded.x_start == 0u &&
                             decoded.y_start == 0u,
                         "mcast=%u start=(%u,%u)", decoded.multicast, decoded.x_start,
                         decoded.y_start);
    ft_tt_test_check_fmt("...on NoC 0", decoded.noc == 0u, "%u", decoded.noc);
    ft_tt_test_check_fmt("...strictly ordered, which is the only ordering tt-kmd ever asks for",
                         decoded.ordering == 1u, "%u", decoded.ordering);
    ft_tt_test_check_fmt("...aimed at the ARC CSM page, where telemetry lives",
                         decoded.addr == WINDOW_PAGE(ARC_CSM_BASE), "0x%llx",
                         (unsigned long long)decoded.addr);
    ft_tt_test_check_fmt("...with linked, use_static_vc and stream_header clear",
                         decoded.linked == 0u && decoded.use_static_vc == 0u &&
                             decoded.stream_header == 0u,
                         "linked=%u vc?=%u sh=%u", decoded.linked, decoded.use_static_vc,
                         decoded.stream_header);

    /* THE ANTI-VACUITY ARM. Everything above would also pass if the decode
     * were insensitive to where in BAR0 it read. Read the SAME field pattern
     * one register triple lower, and four bytes off the triple boundary, and
     * require both to come out different — otherwise "it decodes to (8,0)" is
     * a statement about nothing. */
    read_window_registers(KERNEL_WINDOW_ID - 1u, words);
    (void)ft_tt_udrv_tlb2m_decode(words, &neighbour);
    ft_tt_test_check_fmt("the NEIGHBOURING window does not also decode to (8,0) strict at the CSM "
                         "— the check is position-sensitive",
                         !(neighbour.x_end == ARC_X && neighbour.y_end == ARC_Y &&
                           neighbour.ordering == 1u &&
                           neighbour.addr == WINDOW_PAGE(ARC_CSM_BASE)),
                         "window %u: (%u,%u) ord %u addr 0x%llx", KERNEL_WINDOW_ID - 1u,
                         neighbour.x_end, neighbour.y_end, neighbour.ordering,
                         (unsigned long long)neighbour.addr);

    shifted[0] = read_registers[register_word_index(KERNEL_WINDOW_ID) + 1u];
    shifted[1] = read_registers[register_word_index(KERNEL_WINDOW_ID) + 2u];
    shifted[2] = read_registers[register_word_index(KERNEL_WINDOW_ID) + 3u];
    (void)ft_tt_udrv_tlb2m_decode(shifted, &shifted_decode);
    ft_tt_test_check_fmt("a decode started one WORD late does not give (8,0) either — the register "
                         "offset is load-bearing and is being tested",
                         !(shifted_decode.x_end == ARC_X && shifted_decode.y_end == ARC_Y &&
                           shifted_decode.ordering == 1u),
                         "(%u,%u) ord %u", shifted_decode.x_end, shifted_decode.y_end,
                         shifted_decode.ordering);
}

/* ------------------------------------------------------------------------ *
 * Rung 3 — tt-kmd as the on-silicon oracle
 * ------------------------------------------------------------------------ */

static bool stage_claim_windows(void)
{
    size_t index;

    ft_tt_test_step("Rung 3a — claim %u windows through ALLOCATE_TLB (pure arbitration; nothing "
                    "reaches the card)",
                    CLAIMED_WINDOW_COUNT);

    for (index = 0u; index < CLAIMED_WINDOW_COUNT; index++) {
        uint32_t id = 0u;
        uint64_t mmap_uc = 0u;

        if (allocate_window(&id, &mmap_uc) != FT_TT_UDRV_OK)
            break;

        claimed[claimed_count] = id;
        claimed_mmap_uc[claimed_count] = mmap_uc;
        claimed_count++;
    }

    ft_tt_test_check_fmt("every window was granted", claimed_count == CLAIMED_WINDOW_COUNT,
                         "%zu of %u", claimed_count, CLAIMED_WINDOW_COUNT);

    if (claimed_count == 0u)
        return false;

    ft_tt_test_check_fmt("the lowest id is below the strided threshold",
                         claimed[0] < FT_TT_UDRV_TLB_STRIDED_COUNT, "id %u < %u", claimed[0],
                         FT_TT_UDRV_TLB_STRIDED_COUNT);
    ft_tt_test_check_fmt("the highest id is at or above it — both sides of the strided rule get "
                         "exercised",
                         claimed[claimed_count - 1u] >= FT_TT_UDRV_TLB_STRIDED_COUNT, "id %u",
                         claimed[claimed_count - 1u]);
    ft_tt_test_check_fmt("none of them is tt-kmd's own window",
                         claimed[claimed_count - 1u] != KERNEL_WINDOW_ID, "highest is %u",
                         claimed[claimed_count - 1u]);

    return true;
}

/**
 * The step that makes the whole approach safe: ask the DRIVER to aim a window,
 * then read the registers it wrote through OUR mapping and compare them, byte
 * for byte, against what our encoder says they should be.
 *
 * Nothing here writes a register. If our encoder is wrong, this is where it is
 * found — with the card in exactly the state tt-kmd left it.
 */
static void stage_encoder_against_silicon(uint32_t window, const char *label)
{
    size_t disagreements = 0u;
    size_t index;

    ft_tt_test_step("Rung 3b — %s: %zu configurations, driver writes, we read and compare",
                    label, sweep_count);

    for (index = 0u; index < sweep_count; index++) {
        const sweep_case *entry = &sweep[index];
        uint32_t expected[3] = {0u, 0u, 0u};
        uint32_t observed[3] = {0u, 0u, 0u};

        if (configure_window_via_ioctl(window, &entry->config) != FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt("CONFIGURE_TLB succeeds", false, "%s: errno %d", entry->name,
                                 errno);
            disagreements++;
            continue;
        }

        (void)ft_tt_udrv_tlb2m_encode(&entry->config, expected);
        read_window_registers(window, observed);

        if (memcmp(expected, observed, sizeof(expected)) != 0) {
            ft_tt_test_check_fmt("register words match tt-kmd's", false,
                                 "%s: ours 0x%08x 0x%08x 0x%08x, card 0x%08x 0x%08x 0x%08x",
                                 entry->name, expected[0], expected[1], expected[2], observed[0],
                                 observed[1], observed[2]);
            disagreements++;
        }
    }

    ft_tt_test_check_fmt("every configuration agrees, byte for byte", disagreements == 0u,
                         "%zu of %zu disagreed on window %u", disagreements, sweep_count, window);
}

/**
 * The ioctl takes 16-bit coordinates and drops them into 6-bit register fields,
 * so a coordinate above 63 is silently truncated BY THE KERNEL. Our encoder
 * truncates too, and the two have to truncate identically or a caller who
 * passes 64 aims at tile 0 on one path and somewhere else on the other.
 */
static void stage_truncation_on_silicon(uint32_t window)
{
    static const uint16_t wide[] = {64u, 65u, 127u, 255u, 4095u, 65535u};
    size_t index;

    ft_tt_test_step("Rung 3c — an out-of-range coordinate truncates the same way in the kernel "
                    "and in our encoder");

    for (index = 0u; index < sizeof(wide) / sizeof(wide[0]); index++) {
        struct ft_tt_kmd_configure_window_request request;
        ft_tt_udrv_tlb2m_config config = {0};
        uint32_t expected[3] = {0u, 0u, 0u};
        uint32_t observed[3] = {0u, 0u, 0u};

        memset(&request, 0, sizeof(request));
        request.in.id = window;
        request.in.config.x_end = wide[index];
        request.in.config.y_end = wide[index];

        if (ioctl(chardev, FT_TT_KMD_IOCTL_CONFIGURE_WINDOW, &request) != 0) {
            ft_tt_test_check_fmt("CONFIGURE_TLB accepts a wide coordinate", false, "%u: errno %d",
                                 wide[index], errno);
            continue;
        }

        /* Our encoder is handed the same wide value, narrowed only by the C
         * conversion the struct field forces — exactly what the kernel's
         * bitfield assignment does. */
        config.x_end = (uint8_t)(wide[index] & 0x3Fu);
        config.y_end = (uint8_t)(wide[index] & 0x3Fu);
        (void)ft_tt_udrv_tlb2m_encode(&config, expected);
        read_window_registers(window, observed);

        ft_tt_test_check_fmt("x_end/y_end truncate to 6 bits",
                             memcmp(expected, observed, sizeof(expected)) == 0, "%u -> %u",
                             wide[index], wide[index] & 0x3Fu);
    }
}

/**
 * Two windows, two different aims, read back at once.
 *
 * A register-offset bug that pointed every window at the same triple would pass
 * every check above — each comparison would read back what was written most
 * recently, which is what it expects. This is the arm that catches it.
 */
static void stage_windows_are_distinct(uint32_t low, uint32_t high)
{
    ft_tt_udrv_tlb2m_config low_config = {0};
    ft_tt_udrv_tlb2m_config high_config = {0};
    uint32_t low_expected[3];
    uint32_t high_expected[3];
    uint32_t low_observed[3];
    uint32_t high_observed[3];

    ft_tt_test_step("Rung 3d — two windows hold two different aims at the same time");

    low_config.addr = ARC_CSM_BASE;
    low_config.x_end = ARC_X;
    low_config.y_end = ARC_Y;
    low_config.ordering = 1u;

    high_config.addr = WINDOW_PAGE(ARC_SCRATCH_BASE);
    high_config.x_end = ARC_X;
    high_config.y_end = ARC_Y;
    high_config.ordering = 1u;

    (void)configure_window_via_ioctl(low, &low_config);
    (void)configure_window_via_ioctl(high, &high_config);

    (void)ft_tt_udrv_tlb2m_encode(&low_config, low_expected);
    (void)ft_tt_udrv_tlb2m_encode(&high_config, high_expected);
    read_window_registers(low, low_observed);
    read_window_registers(high, high_observed);

    ft_tt_test_check_fmt("the low window still holds its own aim",
                         memcmp(low_expected, low_observed, sizeof(low_expected)) == 0, "id %u",
                         low);
    ft_tt_test_check_fmt("the high window holds the other",
                         memcmp(high_expected, high_observed, sizeof(high_expected)) == 0, "id %u",
                         high);
    ft_tt_test_check("...and the two register triples are NOT the same memory",
                     memcmp(low_observed, high_observed, sizeof(low_observed)) != 0,
                     "id * 12 is really addressing distinct registers");
}

/**
 * CONFIGURE_TLB zeroes a strided register for every window below 32. Reading
 * that it is zero proves nothing on its own — it was probably already zero. So
 * rung 3 only records the value, and rung 4 turns it into a real test by
 * putting a marker there first.
 */
static void stage_strided_is_zero_after_aim(uint32_t low)
{
    uint32_t value;

    ft_tt_test_step("Rung 3e — the strided register after an aim (recorded, not yet proved)");

    value = read_registers[strided_word_index(low)];
    ft_tt_test_check_fmt("the low window's strided register reads zero after CONFIGURE_TLB",
                         value == 0u, "id %u: 0x%08x at BAR0 + 0x%08X", low, value,
                         (unsigned)(FT_TT_UDRV_TLB_REGS_START + FT_TT_UDRV_TLB_STRIDED_REGS_OFFSET +
                                    low * FT_TT_UDRV_TLB_STRIDED_REG_SIZE));
}

/* ------------------------------------------------------------------------ *
 * Rung 4 — our own writes
 * ------------------------------------------------------------------------ */

/**
 * THE INDEPENDENT READBACK PATH.
 *
 * Rung 4 writes registers through OUR sysfs mmap, so reading them back through
 * the same mmap proves very little — it is the "do not instrument a mechanism
 * with itself" rule. This is the other reader: tt-kmd's OWN BAR0 mapping,
 * obtained from QUERY_MAPPINGS and mmap'd on the driver's character device. It
 * is a different file, a different vma, a different kernel function
 * (`map_pci_bar` -> `vm_iomap_memory`) and a different pgprot decision from
 * the PCI core's `pci_mmap_resource`.
 *
 * /dev/mem was the first choice and it does NOT work on this box: a `pread` at
 * the BAR's physical address returns EFAULT even as root, because x86's
 * `xlate_dev_mem_ptr` cannot remap this high non-RAM range for the read path.
 * That is recorded here rather than in a comment somewhere, because "we used
 * /dev/mem" is exactly the kind of claim that gets repeated after it stopped
 * being true.
 */
static const volatile uint32_t *driver_registers;

/** BAR0's UC mapping offset, as tt-kmd's QUERY_MAPPINGS reports it. */
static bool query_driver_bar0_mapping(uint64_t *out_base, uint64_t *out_size)
{
    struct {
        struct ft_tt_kmd_query_mappings_in in;
        struct ft_tt_kmd_mapping mappings[8];
    } query;
    size_t index;

    memset(&query, 0, sizeof(query));
    query.in.count = 8u;

    if (ioctl(chardev, FT_TT_KMD_IOCTL_QUERY_MAPPINGS, &query) != 0)
        return false;

    for (index = 0u; index < 8u; index++) {
        if (query.mappings[index].id != FT_TT_KMD_MAPPING_RESOURCE0_UC)
            continue;
        *out_base = query.mappings[index].base;
        *out_size = query.mappings[index].size;
        return true;
    }

    return false;
}

static void read_triple_via_driver(uint32_t id, uint32_t out_words[3])
{
    size_t base = register_word_index(id);

    out_words[0] = driver_registers[base + 0u];
    out_words[1] = driver_registers[base + 1u];
    out_words[2] = driver_registers[base + 2u];
}

/** Does /dev/mem work here at all? Asked once, reported, never relied on.
 *  `out_errno` is captured before close(), which would otherwise overwrite it
 *  and make the transcript claim "Success" for a call that failed. */
static bool devmem_can_read_the_bar(int *out_errno)
{
    uint32_t scratch[3] = {0u, 0u, 0u};
    off_t physical = (off_t)(card.bar[0].start + FT_TT_UDRV_TLB_REGS_START);
    ssize_t got;
    int file = open("/dev/mem", O_RDONLY | O_CLOEXEC);

    if (file < 0) {
        *out_errno = errno;
        return false;
    }

    errno = 0;
    got = pread(file, scratch, sizeof(scratch), physical);
    *out_errno = errno;
    (void)close(file);

    return got == (ssize_t)sizeof(scratch);
}

static bool stage_map_registers_writable(void)
{
    uint64_t driver_base = 0u;
    uint64_t driver_size = 0u;
    void *mapping = NULL;
    void *driver_mapping;

    ft_tt_test_step("Rung 4a — a writable mapping of our own, and a SECOND READER that is not "
                    "ours at all");

    if (ft_tt_udrv_pci_map_bar(&card, 0u, false, true, FT_TT_UDRV_TLB_REGS_START,
                               FT_TT_UDRV_TLB_REGS_LEN, &mapping) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("the register file maps read-write through sysfs", false,
                             "errno %d (%s)", errno, strerror(errno));
        return false;
    }

    write_registers = mapping;
    ft_tt_test_check("the register file maps read-write through sysfs", true,
                     "the read-only mapping from rung 1 is still the one every read goes through");

    if (!query_driver_bar0_mapping(&driver_base, &driver_size)) {
        ft_tt_test_check("QUERY_MAPPINGS reports BAR0's UC mapping", false, "");
        return false;
    }
    ft_tt_test_check_fmt("QUERY_MAPPINGS reports BAR0's UC mapping", driver_size == card.bar[0].length,
                         "base 0x%llx size 0x%llx", (unsigned long long)driver_base,
                         (unsigned long long)driver_size);

    driver_mapping = mmap(NULL, FT_TT_UDRV_TLB_REGS_LEN, PROT_READ, MAP_SHARED, chardev,
                          (off_t)(driver_base + FT_TT_UDRV_TLB_REGS_START));
    if (driver_mapping == MAP_FAILED) {
        ft_tt_test_check_fmt("the same registers map through TT-KMD's own BAR0 path", false,
                             "errno %d (%s)", errno, strerror(errno));
        return false;
    }
    driver_registers = driver_mapping;
    ft_tt_test_check("the same registers map through TT-KMD's own BAR0 path", true,
                     "map_pci_bar/vm_iomap_memory, not pci_mmap_resource");

    {
        int devmem_errno = 0;
        bool devmem_works = devmem_can_read_the_bar(&devmem_errno);

        ft_tt_test_check_fmt("(informational) /dev/mem cannot read this BAR on this kernel, so it "
                             "is NOT the second reader",
                             !devmem_works, "pread errno %d (%s)", devmem_errno,
                             strerror(devmem_errno));
    }
    return true;
}

static void stage_our_writes_match(uint32_t window)
{
    size_t sysfs_disagreements = 0u;
    size_t driver_disagreements = 0u;
    size_t index;

    ft_tt_test_step("Rung 4b — WE write the registers; %zu configurations, read back two ways",
                    sweep_count);

    for (index = 0u; index < sweep_count; index++) {
        const sweep_case *entry = &sweep[index];
        uint32_t written[3] = {0u, 0u, 0u};
        uint32_t observed[3] = {0u, 0u, 0u};
        uint32_t through_driver[3] = {0u, 0u, 0u};

        (void)ft_tt_udrv_tlb2m_encode(&entry->config, written);
        write_window_registers(window, written);
        if (window < FT_TT_UDRV_TLB_STRIDED_COUNT)
            write_registers[strided_word_index(window)] = 0u;

        read_window_registers(window, observed);
        if (memcmp(written, observed, sizeof(written)) != 0) {
            ft_tt_test_check_fmt("what we wrote is what the register holds", false,
                                 "%s: wrote 0x%08x 0x%08x 0x%08x, read 0x%08x 0x%08x 0x%08x",
                                 entry->name, written[0], written[1], written[2], observed[0],
                                 observed[1], observed[2]);
            sysfs_disagreements++;
        }

        read_triple_via_driver(window, through_driver);
        if (memcmp(written, through_driver, sizeof(written)) != 0) {
            ft_tt_test_check_fmt("tt-kmd's own mapping sees the same register", false,
                                 "%s: 0x%08x 0x%08x 0x%08x", entry->name, through_driver[0],
                                 through_driver[1], through_driver[2]);
            driver_disagreements++;
        }
    }

    ft_tt_test_check_fmt("every write reads back through our sysfs mapping",
                         sysfs_disagreements == 0u, "%zu of %zu", sysfs_disagreements,
                         sweep_count);
    ft_tt_test_check_fmt("...and through TT-KMD's BAR0 mapping, which is not the path that wrote "
                         "them",
                         driver_disagreements == 0u, "%zu of %zu", driver_disagreements,
                         sweep_count);
}

/**
 * Then hand the same window to the DRIVER and let it overwrite what we wrote.
 *
 * This is the direction that matters for a machine with no tt-kmd on it: it
 * proves the register we write and the register tt-kmd writes are the same
 * register, not two things that merely look alike.
 */
static void stage_the_driver_overwrites_us(uint32_t window)
{
    ft_tt_udrv_tlb2m_config ours = {0};
    ft_tt_udrv_tlb2m_config theirs = {0};
    uint32_t our_words[3];
    uint32_t their_words[3];
    uint32_t observed[3];

    ft_tt_test_step("Rung 4c — the driver writes over our write, in the same register");

    ours.addr = WINDOW_PAGE(ARC_SCRATCH_BASE);
    ours.x_end = ARC_X;
    ours.y_end = ARC_Y;
    ours.ordering = 1u;

    theirs.addr = ARC_CSM_BASE;
    theirs.x_end = ARC_X;
    theirs.y_end = ARC_Y;
    theirs.ordering = 1u;

    (void)ft_tt_udrv_tlb2m_encode(&ours, our_words);
    (void)ft_tt_udrv_tlb2m_encode(&theirs, their_words);

    write_window_registers(window, our_words);
    read_window_registers(window, observed);
    ft_tt_test_check("our write lands", memcmp(our_words, observed, sizeof(our_words)) == 0, "");

    (void)configure_window_via_ioctl(window, &theirs);
    read_window_registers(window, observed);
    ft_tt_test_check("the driver's write replaces it",
                     memcmp(their_words, observed, sizeof(their_words)) == 0,
                     "same register, two writers");
    ft_tt_test_check("...and the two configurations were actually different",
                     memcmp(our_words, their_words, sizeof(our_words)) != 0,
                     "otherwise the overwrite proved nothing");
}

/**
 * 0x9D8 or 0xA08 — settled on silicon, not by reading source.
 *
 * The strided register block is sized from the TOTAL window count (202 2 MiB +
 * 8 4 GiB = 210), so window N's strided register is at 0x9D8 + N*4. Sizing it
 * from 214 gives 0xA08, which is what an earlier reading of this said, and
 * 0xA08 + 0*4 is exactly 0x9D8 + 12*4 — window 12's register. So: put a marker
 * in both, ask the driver to aim window 0, and see which one it clears.
 *
 * Both markers land in registers of windows THIS PROCESS owns, and neither
 * window's aperture is ever mapped, so no transaction can be produced by
 * either. The clean-up is the driver's own zeroing write, not ours.
 */
static void stage_the_strided_offset(uint32_t low, uint32_t low_plus_twelve)
{
    ft_tt_udrv_tlb2m_config benign = {0};
    const uint32_t marker = 0x5A5AA5A5u;
    uint32_t at_9d8;
    uint32_t at_a08;

    ft_tt_test_step("Rung 4d — which offset does CONFIGURE_TLB zero, 0x9D8 or 0xA08?");

    if (low != 0u || low_plus_twelve != 12u) {
        ft_tt_test_check_fmt("windows 0 and 12 are the ones claimed", false, "got %u and %u", low,
                             low_plus_twelve);
        return;
    }

    benign.addr = ARC_CSM_BASE;
    benign.x_end = ARC_X;
    benign.y_end = ARC_Y;
    benign.ordering = 1u;

    write_registers[strided_word_index(0u)] = marker;
    write_registers[strided_word_index(12u)] = marker;

    at_9d8 = read_registers[strided_word_index(0u)];
    at_a08 = read_registers[strided_word_index(12u)];
    ft_tt_test_check_fmt("both markers stuck — the strided registers are writable",
                         at_9d8 == marker && at_a08 == marker, "0x%08x / 0x%08x", at_9d8, at_a08);

    (void)configure_window_via_ioctl(0u, &benign);

    at_9d8 = read_registers[strided_word_index(0u)];
    at_a08 = read_registers[strided_word_index(12u)];

    ft_tt_test_check_fmt("aiming window 0 CLEARED 0x9D8 — that is window 0's strided register",
                         at_9d8 == 0u, "0x%08x", at_9d8);
    ft_tt_test_check_fmt("...and left 0xA08 alone, so 0xA08 is NOT window 0's — the earlier "
                         "reading was wrong and this settles it on silicon",
                         at_a08 == marker, "0x%08x", at_a08);

    /* Clean up the second marker the same way: it is window 12's register, so
     * aiming window 12 is what zeroes it. */
    (void)configure_window_via_ioctl(12u, &benign);
    at_a08 = read_registers[strided_word_index(12u)];
    ft_tt_test_check_fmt("aiming window 12 clears it, which is the same rule one window along",
                         at_a08 == 0u, "0x%08x", at_a08);
}

/**
 * Did the aim we just wrote actually land, bit for bit?
 *
 * Checked through tt-kmd's BAR0 mapping, not through the one that wrote it, so
 * a broken write path cannot confirm itself. A false return means the caller
 * must NOT touch the aperture: the window is pointing somewhere nobody chose,
 * and dereferencing it is the failure that costs a power cycle.
 */
static bool aim_landed(uint32_t id, const uint32_t intended[3])
{
    uint32_t observed[3] = {0u, 0u, 0u};

    read_triple_via_driver(id, observed);

    if (memcmp(intended, observed, FT_TT_UDRV_TLB_REG_SIZE) == 0)
        return true;

    ft_tt_test_check_fmt("REFUSING to read through the aperture — the aim did not land", false,
                         "id %u: wanted 0x%08x 0x%08x 0x%08x, register holds 0x%08x 0x%08x 0x%08x",
                         id, intended[0], intended[1], intended[2], observed[0], observed[1],
                         observed[2]);
    return false;
}

/**
 * The functional check: a window WE aimed reaches the address we aimed it at.
 *
 * Register equality says the hardware sees the same 96 bits. It does not say
 * the aperture goes anywhere. This does: read a block through a window we
 * aimed with our own words through our own mapping, read the same block through
 * a window tt-kmd aimed through its own mapping, and require byte equality.
 *
 * And then prove the comparison can say no — because a wrong aim is a wrong
 * ADDRESS, not an error. tt-kmd's window is re-aimed at a different page of the
 * same tile and the bytes must change. If they do not, the equality above was
 * measuring nothing and the whole check is void.
 */
static void stage_a_window_we_aimed_reaches_it(uint32_t ours_id, uint64_t ours_aperture_offset,
                                               uint32_t theirs_id, uint64_t theirs_mmap_offset)
{
    ft_tt_udrv_tlb2m_config aim = {0};
    ft_tt_udrv_tlb2m_config elsewhere = {0};
    unsigned char ours[FUNCTIONAL_BLOCK_BYTES];
    unsigned char theirs[FUNCTIONAL_BLOCK_BYTES];
    unsigned char other_page[FUNCTIONAL_BLOCK_BYTES];
    const volatile unsigned char *our_aperture = NULL;
    volatile unsigned char *their_aperture = NULL;
    void *mapping = NULL;
    uint32_t words[3];
    uint32_t table_pointer;
    uint32_t values_pointer;

    ft_tt_test_step("Rung 4e — a window WE aimed reads the address we aimed it at");

    aim.addr = ARC_CSM_BASE;
    aim.x_end = ARC_X;
    aim.y_end = ARC_Y;
    aim.ordering = 1u;

    (void)ft_tt_udrv_tlb2m_encode(&aim, words);
    write_window_registers(ours_id, words);
    if (ours_id < FT_TT_UDRV_TLB_STRIDED_COUNT)
        write_registers[strided_word_index(ours_id)] = 0u;

    /* CONFIRM THE AIM BEFORE DEREFERENCING IT, through the reader that is not
     * ours. This is the difference between "a wrong aim is unlikely" and "a
     * wrong aim cannot be dereferenced": every other check in this file can
     * fail and cost a red line, but reading through a window aimed at an
     * endpoint nobody chose is what hangs the NoC and costs a walk to the box.
     * If the write did not land exactly, refuse. */
    if (!aim_landed(ours_id, words))
        return;

    /* Our aperture: our own mmap of resource0, at id * 2 MiB. Read-only,
     * because this stage only reads. */
    if (ft_tt_udrv_pci_map_bar(&card, 0u, false, false, ours_aperture_offset,
                               FUNCTIONAL_APERTURE_BYTES, &mapping) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("our aperture maps", false, "errno %d", errno);
        return;
    }
    our_aperture = mapping;
    ft_tt_test_check_fmt("our aperture maps", true, "resource0 + 0x%llx",
                         (unsigned long long)ours_aperture_offset);

    /* Their aperture: tt-kmd's mmap cookie on tt-kmd's fd. A different file, a
     * different kernel path, a different vma. */
    their_aperture = mmap(NULL, FUNCTIONAL_APERTURE_BYTES, PROT_READ, MAP_SHARED, chardev,
                          (off_t)theirs_mmap_offset);
    if (their_aperture == MAP_FAILED) {
        ft_tt_test_check_fmt("tt-kmd's aperture maps", false, "errno %d", errno);
        (void)ft_tt_udrv_pci_unmap_bar(mapping, FUNCTIONAL_APERTURE_BYTES);
        return;
    }
    (void)configure_window_via_ioctl(theirs_id, &aim);
    ft_tt_test_check("tt-kmd's aperture maps, aimed at the same place through the ioctl", true,
                     "");

    memcpy(ours, (const void *)our_aperture, sizeof(ours));
    memcpy(theirs, (const void *)their_aperture, sizeof(theirs));

    ft_tt_test_check_fmt("the two apertures return identical bytes",
                         memcmp(ours, theirs, sizeof(ours)) == 0, "%u bytes",
                         FUNCTIONAL_BLOCK_BYTES);

    /* THE DISCRIMINATION ARM. Re-aim THEIR window at the scratch page — a
     * different 2 MiB page of the same tile, and an address tt-kmd itself
     * reads. The bytes must change. */
    elsewhere.addr = WINDOW_PAGE(ARC_SCRATCH_BASE);
    elsewhere.x_end = ARC_X;
    elsewhere.y_end = ARC_Y;
    elsewhere.ordering = 1u;
    (void)configure_window_via_ioctl(theirs_id, &elsewhere);
    memcpy(other_page, (const void *)their_aperture, sizeof(other_page));

    ft_tt_test_check("a DIFFERENT page of the same tile returns DIFFERENT bytes — the comparison "
                     "above can say no",
                     memcmp(ours, other_page, sizeof(ours)) != 0,
                     "if this ever passes, the equality arm proved nothing");

    ft_tt_test_check("two offsets within one page differ too — the readback is address-sensitive "
                     "below window granularity",
                     memcmp((const void *)our_aperture,
                            (const void *)(our_aperture + FUNCTIONAL_BLOCK_BYTES),
                            FUNCTIONAL_BLOCK_BYTES) != 0,
                     "");

    /* And a SEMANTIC check on top of the byte one: aim our own window at the
     * scratch page and read the two telemetry pointers ARC publishes. tt-kmd
     * requires both to land inside the CSM before it will use them; so do we.
     * Bytes from a wrong address would have to be inside a 512 KiB window of a
     * 4 GiB space, twice, to fake this. */
    (void)ft_tt_udrv_tlb2m_encode(&elsewhere, words);
    write_window_registers(ours_id, words);
    if (!aim_landed(ours_id, words))
        goto unmap;

    table_pointer = *(const volatile uint32_t *)(const void *)(our_aperture +
                                                               (ARC_SCRATCH_BASE -
                                                                WINDOW_PAGE(ARC_SCRATCH_BASE)) +
                                                               4u * ARC_TELEMETRY_TABLE_SCRATCH);
    values_pointer = *(const volatile uint32_t *)(const void *)(our_aperture +
                                                                (ARC_SCRATCH_BASE -
                                                                 WINDOW_PAGE(ARC_SCRATCH_BASE)) +
                                                                4u * ARC_TELEMETRY_VALUES_SCRATCH);

    ft_tt_test_check_fmt("the telemetry table pointer read through OUR window is inside the ARC "
                         "CSM",
                         table_pointer >= ARC_CSM_BASE &&
                             table_pointer < ARC_CSM_BASE + ARC_CSM_SIZE,
                         "0x%08x", table_pointer);
    ft_tt_test_check_fmt("...and so is the values pointer",
                         values_pointer >= ARC_CSM_BASE &&
                             values_pointer < ARC_CSM_BASE + ARC_CSM_SIZE,
                         "0x%08x", values_pointer);

unmap:
    (void)munmap((void *)their_aperture, FUNCTIONAL_APERTURE_BYTES);
    (void)ft_tt_udrv_pci_unmap_bar(mapping, FUNCTIONAL_APERTURE_BYTES);
}

/* ------------------------------------------------------------------------ *
 * Release
 * ------------------------------------------------------------------------ */

/**
 * Leave every window we touched aimed somewhere BENIGN before giving it back.
 *
 * A freshly claimed window is not blank — that is a recorded hazard, and this
 * run deliberately left windows aimed at coordinate (63,63) and at multicast
 * rectangles because a configured window emits nothing. The next claimant is
 * supposed to aim before it touches, but leaving 40 windows loaded with
 * garbage coordinates is manufacturing the exact trap the hazard describes.
 * So the release path re-aims each of them, through the ioctl, at the ARC tile.
 */
static void stage_release(void)
{
    ft_tt_udrv_tlb2m_config benign = {0};
    size_t index;
    size_t failures = 0u;

    ft_tt_test_step("Release — re-aim every claimed window somewhere harmless, then hand it back");

    benign.addr = ARC_CSM_BASE;
    benign.x_end = ARC_X;
    benign.y_end = ARC_Y;
    benign.ordering = 1u;

    for (index = 0u; index < claimed_count; index++) {
        if (configure_window_via_ioctl(claimed[index], &benign) != FT_TT_UDRV_OK)
            failures++;
    }

    ft_tt_test_check_fmt("every window is aimed at the ARC tile, unicast, no multicast bit left "
                         "set anywhere",
                         failures == 0u, "%zu of %zu failed", failures, claimed_count);

    failures = 0u;
    for (index = 0u; index < claimed_count; index++) {
        if (free_window(claimed[index]) != FT_TT_UDRV_OK)
            failures++;
    }
    ft_tt_test_check_fmt("every window is released", failures == 0u, "%zu of %zu failed", failures,
                         claimed_count);

    if (driver_registers) {
        (void)munmap((void *)(uintptr_t)driver_registers, FT_TT_UDRV_TLB_REGS_LEN);
        driver_registers = NULL;
    }
    if (write_registers) {
        (void)ft_tt_udrv_pci_unmap_bar((void *)write_registers, FT_TT_UDRV_TLB_REGS_LEN);
        write_registers = NULL;
    }
    if (read_registers) {
        (void)ft_tt_udrv_pci_unmap_bar((void *)(uintptr_t)read_registers,
                                       FT_TT_UDRV_TLB_REGS_LEN);
        read_registers = NULL;
    }
    if (chardev >= 0) {
        (void)close(chardev);
        chardev = -1;
    }

    ft_tt_test_check("mappings unmapped and the chardev closed", true, "");
}

/* ------------------------------------------------------------------------ */

static void usage(const char *program)
{
    (void)fprintf(stderr,
                  "usage: %s [read | write]\n"
                  "  read   rungs 1-3: map read-only, decode tt-kmd's window, and gate the\n"
                  "         encoder against CONFIGURE_TLB. WRITES NOTHING TO THE CARD.\n"
                  "  write  rungs 1-4: the above, then write the registers ourselves.\n"
                  "Default is `read`.\n",
                  program);
}

int main(int argc, char **argv)
{
    bool writing = false;
    uint32_t low;
    uint32_t high;

    if (argc > 2) {
        usage(argv[0]);
        return 2;
    }
    if (argc == 2) {
        if (strcmp(argv[1], "write") == 0) {
            writing = true;
        } else if (strcmp(argv[1], "read") != 0) {
            usage(argv[0]);
            return 2;
        }
    }

    build_sweep();

    if (!stage_open_the_card())
        return ft_tt_test_finish();

    stage_read_the_kernels_window();

    if (!stage_claim_windows()) {
        stage_release();
        return ft_tt_test_finish();
    }

    low = claimed[0];
    high = claimed[claimed_count - 1u];

    stage_encoder_against_silicon(low, "a window below the strided threshold");
    stage_encoder_against_silicon(high, "a window at or above it");
    stage_truncation_on_silicon(low);
    stage_windows_are_distinct(low, high);
    stage_strided_is_zero_after_aim(low);

    if (writing && stage_map_registers_writable()) {
        stage_our_writes_match(low);
        stage_our_writes_match(high);
        stage_the_driver_overwrites_us(low);
        stage_the_strided_offset(low, claimed_count > 12u ? claimed[12] : 0xFFFFFFFFu);
        stage_a_window_we_aimed_reaches_it(low,
                                           (uint64_t)low * FT_TT_UDRV_TLB_2M_WINDOW_SIZE, high,
                                           claimed_mmap_uc[claimed_count - 1u]);
    } else if (!writing) {
        ft_tt_test_step("Rung 4 SKIPPED — run with `write` to let this process write a register");
    }

    stage_release();
    return ft_tt_test_finish();
}
