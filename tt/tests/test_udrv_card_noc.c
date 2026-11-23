/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_card_noc.c
 * @brief ON-SILICON bring-up for the userspace driver's NoC TRANSPORT — driver
 *   step 5. Windows we aim ourselves, apertures we map ourselves, bytes moved
 *   with no ioctl anywhere in the data path, and tt-kmd used as the oracle for
 *   every one of them WHILE IT IS STILL HERE TO BE ONE.
 *
 *   Three questions this file exists to answer, in the order they are asked:
 *
 *     1. Does a window WE aimed reach the same endpoint as one tt-kmd aimed?
 *        Byte comparison through two apertures, and a discrimination arm that
 *        must fail, because a wrong aim is a wrong ADDRESS and not an error.
 *     2. DOES A WRITE-COMBINED STORE REACH THE CARD? Nothing has ever written
 *        through `resource0_wc` — mapping it is not using it — and it is the
 *        bulk upload path, so this is the single most important new fact of the
 *        round. Answered twice, at two different levels, and BOTH read back
 *        through a path that is not the one that wrote them.
 *     3. `stream_header` and the 3-bit `static_vc` cannot be expressed by
 *        CONFIGURE_TLB, so the driver cannot oracle them and they have never
 *        been swept. Swept here at the register level, where the sweep is free
 *        of risk, and functionally only where an oracle exists.
 *
 *   WHY THE SWEEP IS SAFE. Configuring a window emits NO NoC transaction; only
 *   touching its 2 MiB aperture does. Every register-level arm below therefore
 *   writes configurations whose apertures are never mapped, which is what makes
 *   it safe to put unreviewed bits in a NoC routing field at all. The arms that
 *   DO dereference an aperture use only endpoints derived from live ARC
 *   telemetry through ft_tt_coord, and only cores this process has claimed.
 *
 *   WHY A BAD AIM IS DETECTABLE AND NOT MERELY UNLIKELY. `ft_tt_udrv_noc_aim()`
 *   reads its own register triple back and refuses, with nothing dereferenced,
 *   unless all three words are bit-identical to the intent. This file goes one
 *   step further while it can: it re-reads the same triple through TT-KMD's
 *   BAR0 mapping — a different file, a different vma, a different kernel
 *   function — and refuses to touch the aperture unless that agrees too. That
 *   second reader disappears with the module, which is exactly why the
 *   comparison has to be made on this machine.
 *
 * @status Runs on the p150a at 0000:12:00.0, with tt-kmd
 *   loaded. Needs root (the BAR sysfs files are 0600 root:root).
 * @issues
 *   - The functional static_vc arm covers VC 0 only. See stage_static_vc_sweep().
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include "ft_tt_coord.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_kmd_abi.h"
#include "ft_tt_memmap.h"
#include "ft_tt_noc.h"
#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_noc.h"
#include "ft_tt_udrv_pci_sysfs.h"
#include "ft_tt_udrv_tlb_encode.h"

#define CHARDEV_PATH "/dev/tenstorrent/0"

/** The window tt-kmd claims for itself. */
#define KERNEL_WINDOW_ID (FT_TT_UDRV_TLB_2M_WINDOW_COUNT - 1u)

/** ARC tile, TRANSLATED. The one coordinate live hardware has confirmed, and
 *  the only tile the register-level arms ever name. */
#define ARC_X 8u
#define ARC_Y 0u
#define ARC_CSM_BASE 0x10000000ull
#define ARC_SCRATCH_BASE 0x80030400ull
#define WINDOW_PAGE(addr) ((addr) & ~((uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE - 1u))

/** Two 64 KiB scratch regions in L1, far above FT_TT_MEM_MAP_END (0x9F80) so
 *  nothing firmware owns is in range, and far below 1536 KiB so the span is
 *  wholly inside L1. */
#define L1_SCRATCH_WC 0x00040000ull
#define L1_SCRATCH_UC 0x00060000ull
#define PAYLOAD_BYTES (64u * 1024u)

/** A DRAM span deliberately chosen to STRADDLE a 2 MiB window boundary, so the
 *  window walk is exercised on silicon and not only in the off-hardware gate.
 *  64 B aligned, which is the DRAM read alignment. */
#define DRAM_SCRATCH_BASE 0x101FF000ull
#define DRAM_SPAN_BYTES (16u * 1024u)

/** How many windows this run claims through tt-kmd's arbitrator. */
#define CLAIMED_WINDOW_COUNT 6u

/* ------------------------------------------------------------------------ *
 * Session state
 * ------------------------------------------------------------------------ */

static ft_tt_udrv_pci_device card;
static ft_tt_udrv_noc_device udrv;
static bool udrv_open;

static int chardev = -1;
static const volatile uint32_t *driver_registers; /**< tt-kmd's own BAR0 mapping */

static ft_tt_hardware_session session;
static bool session_open;
static uint16_t core_a_x, core_a_y;
static uint16_t core_b_x, core_b_y;

static uint32_t claimed[CLAIMED_WINDOW_COUNT];
static size_t claimed_count;

/** Windows this run uses, taken from `claimed` once it is populated. */
static uint32_t window_ours_wc;
static uint32_t window_ours_uc;
static uint32_t window_theirs;
static uint32_t window_register_probe;
static uint32_t window_sweep;

/* ------------------------------------------------------------------------ *
 * The fixture pattern — non-periodic, and checked to be
 * ------------------------------------------------------------------------ */

/**
 * A pattern that repeats at any power of two would make every byte comparison
 * below vacuous: an upload that landed one block late, or in the wrong 4 KiB
 * page, would still compare equal. That exact failure was found here once with
 * a 256-periodic fixture, so the generator is an LCG over the ABSOLUTE offset
 * and its non-periodicity is a check in `test_noc_copy.c`, not a comment.
 */
static unsigned char pattern_byte(size_t offset, uint32_t salt)
{
    uint32_t state = (uint32_t)(offset + 1u) * 2654435761u + salt;

    state ^= state >> 15;
    state *= 2246822519u;
    state ^= state >> 13;
    return (unsigned char)(state & 0xFFu);
}

static void fill_pattern(unsigned char *buffer, size_t bytes, uint32_t salt)
{
    size_t index;

    for (index = 0u; index < bytes; index++)
        buffer[index] = pattern_byte(index, salt);
}

static double seconds_since(const struct timespec *start)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - start->tv_sec) + (double)(now.tv_nsec - start->tv_nsec) / 1e9;
}

/* ------------------------------------------------------------------------ *
 * tt-kmd, as arbitrator and as second reader
 * ------------------------------------------------------------------------ */

static int allocate_window(uint32_t *out_id)
{
    struct ft_tt_kmd_allocate_window_request request;

    memset(&request, 0, sizeof(request));
    request.in.size = (uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE;

    if (ioctl(chardev, FT_TT_KMD_IOCTL_ALLOCATE_WINDOW, &request) != 0)
        return FT_TT_UDRV_EIO;

    *out_id = request.out.id;
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

static bool query_driver_bar0_mapping(uint64_t *out_base)
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
        return true;
    }
    return false;
}

/** Window `id`'s register triple, read through TT-KMD's BAR0 mapping. */
static void read_triple_via_driver(uint32_t id, uint32_t out_words[3])
{
    size_t base = (size_t)id * (FT_TT_UDRV_TLB_REG_SIZE / 4u);

    out_words[0] = driver_registers[base + 0u];
    out_words[1] = driver_registers[base + 1u];
    out_words[2] = driver_registers[base + 2u];
}

/**
 * Aim one of OUR windows and confirm the aim through the reader that is not
 * ours, before anything is allowed to dereference the aperture.
 *
 * This is the difference between "a wrong aim is unlikely" and "a wrong aim
 * cannot be dereferenced". Every other check in this file can fail and cost a
 * red line; reading or writing through a window aimed at an endpoint nobody
 * chose is what hangs the NoC and costs a walk to the box.
 */
static bool aim_ours_verified(uint32_t id, const ft_tt_udrv_tlb2m_config *config)
{
    uint32_t intended[3] = {0u, 0u, 0u};
    uint32_t through_driver[3] = {0u, 0u, 0u};
    int rc;

    rc = ft_tt_udrv_noc_window_aim(&udrv, id, config);
    if (rc != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("our aim landed (self-verified readback)", false, "window %u rc=%d",
                             id, rc);
        return false;
    }

    (void)ft_tt_udrv_tlb2m_encode(config, intended);
    read_triple_via_driver(id, through_driver);
    if (memcmp(intended, through_driver, sizeof(intended)) != 0) {
        ft_tt_test_check_fmt("REFUSING the aperture — TT-KMD's mapping does not see our aim",
                             false, "window %u: wanted 0x%08x 0x%08x 0x%08x, holds 0x%08x 0x%08x "
                                    "0x%08x",
                             id, intended[0], intended[1], intended[2], through_driver[0],
                             through_driver[1], through_driver[2]);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung 1 — the card, our register mapping, and the second reader
 * ------------------------------------------------------------------------ */

static bool stage_open_the_card(bool writing)
{
    size_t count = 0u;
    uint64_t driver_base = 0u;
    void *driver_mapping;

    ft_tt_test_step("Rung 1 — the card, BAR0's register file through sysfs, and TT-KMD as the "
                    "second reader");

    ft_tt_test_check_fmt("running as root (the BAR files are 0600 root:root)", geteuid() == 0u,
                         "euid=%u", (unsigned)geteuid());

    if (ft_tt_udrv_pci_find_devices(NULL, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT,
                                    FT_TT_UDRV_PCI_DEVICE_BLACKHOLE, &card, 1u,
                                    &count) != FT_TT_UDRV_OK ||
        count == 0u) {
        ft_tt_test_check("a Blackhole is present in sysfs", false, "none found");
        return false;
    }
    ft_tt_test_check_fmt("a Blackhole is present in sysfs", true, "%s", card.address);

    if (ft_tt_udrv_noc_device_open(&card, writing, &udrv) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("BAR0's TLB register file maps", false, "errno %d (%s)", errno,
                             strerror(errno));
        return false;
    }
    udrv_open = true;
    ft_tt_test_check_fmt("BAR0's TLB register file maps, uncached", true, "resource0 + 0x%08X, %s",
                         FT_TT_UDRV_TLB_REGS_START, writing ? "PROT_READ|PROT_WRITE" : "PROT_READ");

    chardev = open(CHARDEV_PATH, O_RDWR | O_CLOEXEC);
    if (chardev < 0) {
        ft_tt_test_check_fmt("tt-kmd's chardev opens — it is the ORACLE and the way back", false,
                             "%s errno %d", CHARDEV_PATH, errno);
        return false;
    }
    ft_tt_test_check_fmt("tt-kmd's chardev opens — it is the ORACLE and the way back", true,
                         "fd=%d", chardev);

    if (!query_driver_bar0_mapping(&driver_base)) {
        ft_tt_test_check("QUERY_MAPPINGS reports BAR0's UC mapping", false, "");
        return false;
    }
    driver_mapping = mmap(NULL, FT_TT_UDRV_TLB_REGS_LEN, PROT_READ, MAP_SHARED, chardev,
                          (off_t)(driver_base + FT_TT_UDRV_TLB_REGS_START));
    if (driver_mapping == MAP_FAILED) {
        ft_tt_test_check_fmt("the same registers map through TT-KMD's own BAR0 path", false,
                             "errno %d", errno);
        return false;
    }
    driver_registers = driver_mapping;
    ft_tt_test_check("the same registers map through TT-KMD's own BAR0 path", true,
                     "map_pci_bar/vm_iomap_memory, not pci_mmap_resource");

    /* THE SECOND READER IS ONLY A SECOND READER IF IT AGREES. Compare all 606
     * configuration words, both mappings, before anything leans on either. */
    {
        size_t disagreements = 0u;
        size_t all_ones = 0u;
        uint32_t id;

        for (id = 0u; id < FT_TT_UDRV_TLB_2M_WINDOW_COUNT; id++) {
            uint32_t ours[3] = {0u, 0u, 0u};
            uint32_t theirs[3] = {0u, 0u, 0u};

            (void)ft_tt_udrv_noc_window_read(&udrv, id, ours);
            read_triple_via_driver(id, theirs);
            if (memcmp(ours, theirs, sizeof(ours)) != 0)
                disagreements++;
            if (ours[0] == 0xFFFFFFFFu && ours[1] == 0xFFFFFFFFu && ours[2] == 0xFFFFFFFFu)
                all_ones++;
        }
        ft_tt_test_check_fmt("our mapping and TT-KMD's read the same 606 words",
                             disagreements == 0u, "%zu of %u windows disagreed", disagreements,
                             FT_TT_UDRV_TLB_2M_WINDOW_COUNT);
        ft_tt_test_check_fmt("no window reads all-ones — the aperture decodes, it does not float",
                             all_ones == 0u, "%zu of %u", all_ones,
                             FT_TT_UDRV_TLB_2M_WINDOW_COUNT);
    }
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung 2 — coordinates from live telemetry, and the cores claimed
 * ------------------------------------------------------------------------ */

static bool stage_open_the_session(void)
{
    uint32_t indices[2] = {0u, 1u};
    uint32_t core_count;

    ft_tt_test_step("Rung 2 — coordinates DERIVED from live ARC telemetry, and the two cores "
                    "claimed");

    if (ft_tt_hardware_session_open(&session) != FT_TT_OK) {
        ft_tt_test_check("the tt-kmd session opens — it is the oracle transport", false, "");
        return false;
    }
    session_open = true;

    /* The session now honours FT_TT_DRIVER, and this suite's whole value is that
     * the readback comes through a mapping that is NOT the one under test. An
     * environment that redirected the oracle onto the userspace driver would make
     * every comparison here self-instrumenting and every check pass for free. */
    if (!ft_tt_hardware_session_kmd_context(&session)) {
        ft_tt_test_check("the ORACLE session is on tt-kmd — FT_TT_DRIVER must not redirect it",
                         false, "the second reader would be the driver under test");
        return false;
    }

    core_count = ft_tt_coord_tensix_core_count(&session.grid);
    ft_tt_test_check_fmt("the grid reports its own core count — nothing is sized from 140",
                         core_count >= 2u, "%u enabled Tensix", core_count);

    if (ft_tt_hardware_session_claim_cores(&session, indices, 2u, "udrv noc transport bring-up",
                                           NULL, NULL) != FT_TT_OK) {
        ft_tt_test_check("cores 0 and 1 are claimed exclusively", false,
                         "another process holds one; see stderr");
        return false;
    }
    ft_tt_test_check("cores 0 and 1 are claimed exclusively", true, "");

    if (ft_tt_hardware_session_tensix(&session, 0u, &core_a_x, &core_a_y) != FT_TT_OK ||
        ft_tt_hardware_session_tensix(&session, 1u, &core_b_x, &core_b_y) != FT_TT_OK) {
        ft_tt_test_check("two TRANSLATED Tensix coordinates resolve", false, "");
        return false;
    }
    ft_tt_test_check_fmt("two TRANSLATED Tensix coordinates resolve, and they are DIFFERENT — the "
                         "coordinate-sensitivity arms need that",
                         !(core_a_x == core_b_x && core_a_y == core_b_y), "A=(%u,%u) B=(%u,%u)",
                         core_a_x, core_a_y, core_b_x, core_b_y);
    return true;
}

static bool stage_claim_windows(void)
{
    size_t index;

    ft_tt_test_step("Rung 2b — claim %u windows (pure arbitration; nothing reaches the card)",
                    CLAIMED_WINDOW_COUNT);

    for (index = 0u; index < CLAIMED_WINDOW_COUNT; index++) {
        uint32_t id = 0u;

        if (allocate_window(&id) != FT_TT_UDRV_OK)
            break;
        claimed[claimed_count++] = id;
    }

    ft_tt_test_check_fmt("every window was granted", claimed_count == CLAIMED_WINDOW_COUNT,
                         "%zu of %u", claimed_count, CLAIMED_WINDOW_COUNT);
    if (claimed_count < CLAIMED_WINDOW_COUNT)
        return false;

    window_ours_wc = claimed[0];
    window_ours_uc = claimed[1];
    window_theirs = claimed[2];
    window_register_probe = claimed[3];
    window_sweep = claimed[4];

    ft_tt_test_check_fmt("none of them is tt-kmd's own window", claimed[claimed_count - 1u] !=
                             KERNEL_WINDOW_ID,
                         "kernel window %u; ours %u %u %u %u %u %u", KERNEL_WINDOW_ID, claimed[0],
                         claimed[1], claimed[2], claimed[3], claimed[4], claimed[5]);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung 3 — a window WE aimed reads what one TT-KMD aimed
 * ------------------------------------------------------------------------ */

/**
 * Read the same 4 KiB of the ARC CSM through a window we aimed with our own
 * register writes and a window tt-kmd aimed through CONFIGURE_TLB. Byte
 * equality, then a discrimination arm that MUST fail.
 *
 * The ARC CSM is chosen because it is an address tt-kmd itself reads on that
 * tile every time anything touches telemetry, so neither aperture is pointed
 * anywhere the driver does not already point.
 */
static void stage_read_agreement(void)
{
    ft_tt_udrv_tlb2m_config aim;
    ft_tt_udrv_tlb2m_config elsewhere;
    ft_tt_udrv_noc_path path;
    unsigned char ours[4096];
    unsigned char theirs[4096];
    unsigned char other_page[4096];
    volatile unsigned char *their_aperture;
    uint64_t theirs_mmap = 0u;

    ft_tt_test_step("Rung 3 — a window WE aimed reads the same bytes as one TT-KMD aimed");

    memset(&aim, 0, sizeof(aim));
    aim.addr = WINDOW_PAGE(ARC_CSM_BASE);
    aim.x_end = ARC_X;
    aim.y_end = ARC_Y;
    aim.ordering = 1u;

    if (!aim_ours_verified(window_ours_uc, &aim))
        return;
    ft_tt_test_check("our aim landed, and TT-KMD's mapping agrees bit for bit", true,
                     "the aperture may now be dereferenced");

    if (ft_tt_udrv_noc_path_open(&udrv, window_ours_uc, false, &path) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("our aperture maps", false, "errno %d", errno);
        return;
    }
    memcpy(ours, (const void *)path.aperture, sizeof(ours));

    /* Theirs: tt-kmd's ioctl aim, and tt-kmd's own mmap cookie. A different
     * file, a different kernel path, a different vma. The cookie is
     * QUERY_MAPPINGS' BAR0 base plus id * 2 MiB, which is the same page
     * ALLOCATE_TLB's mmap_offset_uc names. */
    {
        uint64_t driver_base = 0u;

        if (!query_driver_bar0_mapping(&driver_base)) {
            ft_tt_test_check("QUERY_MAPPINGS still answers", false, "");
            (void)ft_tt_udrv_noc_path_close(&path);
            return;
        }
        theirs_mmap = driver_base + (uint64_t)window_theirs * FT_TT_UDRV_TLB_2M_WINDOW_SIZE;
    }

    (void)configure_window_via_ioctl(window_theirs, &aim);
    their_aperture = mmap(NULL, (size_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE, PROT_READ, MAP_SHARED,
                          chardev, (off_t)theirs_mmap);
    if (their_aperture == MAP_FAILED) {
        ft_tt_test_check_fmt("tt-kmd's aperture maps", false, "errno %d", errno);
        (void)ft_tt_udrv_noc_path_close(&path);
        return;
    }
    memcpy(theirs, (const void *)their_aperture, sizeof(theirs));

    ft_tt_test_check_fmt("the two apertures return identical bytes",
                         memcmp(ours, theirs, sizeof(ours)) == 0, "%zu bytes", sizeof(ours));

    /* THE DISCRIMINATION ARM. Re-aim THEIR window at the ARC scratch page — a
     * different 2 MiB page of the same tile, and an address tt-kmd itself
     * reads. The bytes must change; if they do not, the equality above was
     * measuring nothing. */
    memset(&elsewhere, 0, sizeof(elsewhere));
    elsewhere.addr = WINDOW_PAGE(ARC_SCRATCH_BASE);
    elsewhere.x_end = ARC_X;
    elsewhere.y_end = ARC_Y;
    elsewhere.ordering = 1u;
    (void)configure_window_via_ioctl(window_theirs, &elsewhere);
    memcpy(other_page, (const void *)their_aperture, sizeof(other_page));

    ft_tt_test_check("a DIFFERENT page of the same tile returns DIFFERENT bytes — the comparison "
                     "above can say no",
                     memcmp(ours, other_page, sizeof(ours)) != 0, "");
    ft_tt_test_check("two offsets within one page differ too — the readback is address-sensitive "
                     "below window granularity",
                     memcmp((const void *)path.aperture,
                            (const void *)((const volatile unsigned char *)path.aperture + 4096u),
                            4096u) != 0,
                     "");

    (void)munmap((void *)their_aperture, (size_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE);
    (void)ft_tt_udrv_noc_path_close(&path);
}

/* ------------------------------------------------------------------------ *
 * Rung 4a — DOES A WRITE-COMBINED STORE REACH THE CARD?
 * ------------------------------------------------------------------------ */

/**
 * The narrowest possible form of the question, with zero NoC transactions
 * involved: store a TLB configuration triple through a WRITE-COMBINED mapping
 * of `resource0_wc` and read it back through TT-KMD's uncached BAR0 mapping.
 *
 * The target is a configuration register of a window this process owns and
 * whose aperture is never mapped, so nothing can be emitted whatever the bits
 * say. If this passes, a WC store reaches the ASIC. If it fails, every WC
 * upload measurement this project has ever taken went through tt-kmd's mapping
 * and says nothing about ours.
 *
 * The values written are ENCODED CONFIGURATIONS, not arbitrary bit patterns:
 * three of them, chosen so that between them all three words change, and so
 * that a stale readback of any one is distinguishable from a correct readback
 * of another.
 */
static void stage_wc_store_reaches_the_card(void)
{
    ft_tt_udrv_tlb2m_config configs[3];
    void *wc_mapping = NULL;
    volatile uint32_t *wc_registers;
    uint32_t before[3] = {0u, 0u, 0u};
    size_t base = (size_t)window_register_probe * (FT_TT_UDRV_TLB_REG_SIZE / 4u);
    size_t index;
    size_t landed = 0u;
    bool distinct = true;

    ft_tt_test_step("Rung 4a — A WRITE-COMBINED STORE, read back through TT-KMD's mapping. "
                    "Nothing has ever written through resource0_wc");

    if (ft_tt_udrv_pci_map_bar(&card, 0u, true, true, FT_TT_UDRV_TLB_REGS_START,
                               FT_TT_UDRV_TLB_REGS_LEN, &wc_mapping) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("resource0_wc maps read-WRITE", false, "errno %d (%s)", errno,
                             strerror(errno));
        return;
    }
    wc_registers = wc_mapping;
    ft_tt_test_check("resource0_wc maps read-WRITE", true,
                     "pci_mmap_resource with pgprot_writecombine");

    memset(configs, 0, sizeof(configs));
    configs[0].addr = WINDOW_PAGE(ARC_CSM_BASE);
    configs[0].x_end = ARC_X;
    configs[0].y_end = ARC_Y;
    configs[1].addr = WINDOW_PAGE(ARC_SCRATCH_BASE);
    configs[1].x_end = ARC_X;
    configs[1].y_end = ARC_Y;
    configs[1].ordering = 1u;
    configs[2].addr = WINDOW_PAGE(ARC_CSM_BASE);
    configs[2].x_end = ARC_X;
    configs[2].y_end = ARC_Y;
    configs[2].ordering = 3u;
    configs[2].linked = 1u;
    configs[2].use_static_vc = 1u;

    read_triple_via_driver(window_register_probe, before);

    for (index = 0u; index < 3u; index++) {
        uint32_t intended[3] = {0u, 0u, 0u};
        uint32_t observed[3] = {0u, 0u, 0u};

        (void)ft_tt_udrv_tlb2m_encode(&configs[index], intended);

        /* A readback that equals what was ALREADY there proves nothing. Refuse
         * to count this arm unless the value is actually changing. */
        read_triple_via_driver(window_register_probe, observed);
        if (memcmp(intended, observed, sizeof(intended)) == 0)
            distinct = false;

        wc_registers[base + 0u] = intended[0];
        wc_registers[base + 1u] = intended[1];
        wc_registers[base + 2u] = intended[2];
        ft_tt_noc_copy_store_fence();

        read_triple_via_driver(window_register_probe, observed);
        if (memcmp(intended, observed, sizeof(intended)) == 0) {
            landed++;
        } else {
            ft_tt_test_check_fmt("a WC store lands", false,
                                 "config %zu: wrote 0x%08x 0x%08x 0x%08x, TT-KMD reads 0x%08x "
                                 "0x%08x 0x%08x",
                                 index, intended[0], intended[1], intended[2], observed[0],
                                 observed[1], observed[2]);
        }
    }

    ft_tt_test_check_fmt("A WRITE-COMBINED STORE REACHES THE CARD — read back through TT-KMD's "
                         "own BAR0 mapping, which is not the path that wrote it",
                         landed == 3u, "%zu of 3 configurations", landed);
    ft_tt_test_check("...and every one of them CHANGED the register, so the readback is not "
                     "agreeing with what was already there",
                     distinct, "if this fails the arm above is vacuous");

    /* Leave the probe window aimed somewhere harmless. */
    {
        ft_tt_udrv_tlb2m_config benign;

        memset(&benign, 0, sizeof(benign));
        benign.addr = WINDOW_PAGE(ARC_CSM_BASE);
        benign.x_end = ARC_X;
        benign.y_end = ARC_Y;
        benign.ordering = 1u;
        (void)configure_window_via_ioctl(window_register_probe, &benign);
    }
    (void)ft_tt_udrv_pci_unmap_bar(wc_mapping, FT_TT_UDRV_TLB_REGS_LEN);
}

/* ------------------------------------------------------------------------ *
 * Rung 4b — the bulk upload path: a WC store through an APERTURE
 * ------------------------------------------------------------------------ */

/**
 * The question rung 4a cannot answer: a WC store through a window's APERTURE,
 * which becomes a NoC write transaction, into a real endpoint — and read back
 * through TT-KMD's transport, which is a different window, a different mapping
 * and a different aiming mechanism end to end.
 *
 * @param write_combine  which cache mode the upload path uses
 * @param address        L1 scratch address for this arm
 * @param salt           makes this arm's pattern different from every other's
 */
static void upload_and_verify(uint32_t window, bool write_combine, uint64_t address, uint32_t salt,
                              const char *label)
{
    ft_tt_udrv_noc_path path;
    ft_tt_udrv_tlb2m_config aim;
    ft_tt_noc_coord core_a;
    ft_tt_noc_coord core_b;
    unsigned char *sent = malloc(PAYLOAD_BYTES);
    unsigned char *received = malloc(PAYLOAD_BYTES);
    unsigned char *elsewhere = malloc(PAYLOAD_BYTES);
    struct timespec start;
    double elapsed;
    int rc;

    if (!sent || !received || !elsewhere) {
        ft_tt_test_check_fmt("allocation", false, "%s", label);
        goto done;
    }

    core_a.x = core_a_x;
    core_a.y = core_a_y;
    core_b.x = core_b_x;
    core_b.y = core_b_y;

    fill_pattern(sent, PAYLOAD_BYTES, salt);

    /* THE PRE-ARM. Read the target through the ORACLE before writing anything:
     * if it already held the pattern, the comparison afterwards would be a
     * statement about nothing. */
    memset(received, 0, PAYLOAD_BYTES);
    rc = ft_tt_read_l1(&session.runtime.binding.path, core_a, address, received, PAYLOAD_BYTES);
    ft_tt_test_check_fmt("the oracle can read the target before we write it", rc == FT_TT_OK,
                         "%s, rc=%d", label, rc);
    ft_tt_test_check_fmt("...and it does NOT already hold our pattern",
                         memcmp(sent, received, PAYLOAD_BYTES) != 0, "%s", label);

    if (ft_tt_udrv_noc_path_open(&udrv, window, write_combine, &path) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("our aperture maps", false, "%s errno %d", label, errno);
        goto done;
    }

    /* Confirm the aim through TT-KMD before a single byte is stored. The
     * transport verifies it too, but only against the mapping that wrote it. */
    memset(&aim, 0, sizeof(aim));
    aim.addr = WINDOW_PAGE(address);
    aim.x_end = (uint8_t)core_a_x;
    aim.y_end = (uint8_t)core_a_y;
    if (!aim_ours_verified(window, &aim)) {
        (void)ft_tt_udrv_noc_path_close(&path);
        goto done;
    }

    (void)clock_gettime(CLOCK_MONOTONIC, &start);
    rc = ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, address, sent, PAYLOAD_BYTES, false,
                                 FT_TT_NOC_WIDTH_BLOCK16);
    elapsed = seconds_since(&start);

    ft_tt_test_check_fmt("the upload returns OK", rc == FT_TT_UDRV_OK, "%s, rc=%d", label, rc);
    ft_tt_test_check_fmt("...no aim was refused", ft_tt_udrv_noc_path_aim_failures(&path) == 0u,
                         "%s, %llu aims, %llu refused", label,
                         (unsigned long long)ft_tt_udrv_noc_path_aim_count(&path),
                         (unsigned long long)ft_tt_udrv_noc_path_aim_failures(&path));

    /* THE READBACK, THROUGH A PATH THAT IS NOT THE ONE THAT WROTE IT: tt-kmd's
     * window, tt-kmd's mapping, tt-kmd's aim. */
    memset(received, 0, PAYLOAD_BYTES);
    rc = ft_tt_read_l1(&session.runtime.binding.path, core_a, address, received, PAYLOAD_BYTES);
    ft_tt_test_check_fmt("the oracle reads it back", rc == FT_TT_OK, "%s, rc=%d", label, rc);
    ft_tt_test_check_fmt("A STORE THROUGH OUR OWN APERTURE REACHED THE CARD — byte-exact through "
                         "TT-KMD's transport, which is not the path that wrote it",
                         memcmp(sent, received, PAYLOAD_BYTES) == 0, "%s, %u bytes, %.1f MB/s "
                         "host->card",
                         label, PAYLOAD_BYTES,
                         elapsed > 0.0 ? ((double)PAYLOAD_BYTES / elapsed) / 1e6 : 0.0);

    /* ADDRESS SENSITIVITY: the next 64 KiB of the same core must NOT hold it. */
    memset(elsewhere, 0, PAYLOAD_BYTES);
    (void)ft_tt_read_l1(&session.runtime.binding.path, core_a, address + PAYLOAD_BYTES, elsewhere,
                        PAYLOAD_BYTES);
    ft_tt_test_check_fmt("...and the NEXT 64 KiB of the same core does not — the write went where "
                         "it was aimed, not everywhere",
                         memcmp(sent, elsewhere, PAYLOAD_BYTES) != 0, "%s", label);

    /* COORDINATE SENSITIVITY: the same address on the other claimed core must
     * not hold it either. A window aimed at the wrong tile is the failure this
     * catches, and it is silent by construction. */
    memset(elsewhere, 0, PAYLOAD_BYTES);
    (void)ft_tt_read_l1(&session.runtime.binding.path, core_b, address, elsewhere, PAYLOAD_BYTES);
    ft_tt_test_check_fmt("...nor does the SAME address on a different core",
                         memcmp(sent, elsewhere, PAYLOAD_BYTES) != 0, "%s, A=(%u,%u) B=(%u,%u)",
                         label, core_a_x, core_a_y, core_b_x, core_b_y);

    /* AND IT CAN CHANGE. A second upload with a different pattern must be
     * visible; otherwise the first could have been a coincidence of whatever
     * was in L1. */
    fill_pattern(sent, PAYLOAD_BYTES, salt + 977u);
    (void)ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, address, sent, PAYLOAD_BYTES, false,
                                 FT_TT_NOC_WIDTH_BLOCK16);
    memset(received, 0, PAYLOAD_BYTES);
    (void)ft_tt_read_l1(&session.runtime.binding.path, core_a, address, received, PAYLOAD_BYTES);
    ft_tt_test_check_fmt("...a SECOND upload with a different pattern replaces it",
                         memcmp(sent, received, PAYLOAD_BYTES) == 0, "%s", label);

    /* And the transport reads its own writes back — the weakest of the checks,
     * recorded because it is the only one the target board will still have. */
    memset(received, 0, PAYLOAD_BYTES);
    rc = ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, address, received, PAYLOAD_BYTES, true,
                                 FT_TT_NOC_WIDTH_BLOCK16);
    ft_tt_test_check_fmt("...and OUR OWN transport reads it back too (self-instrumenting, and "
                         "recorded as such)",
                         rc == FT_TT_UDRV_OK && memcmp(sent, received, PAYLOAD_BYTES) == 0, "%s",
                         label);

    (void)ft_tt_udrv_noc_path_close(&path);

done:
    free(sent);
    free(received);
    free(elsewhere);
}

static void stage_the_upload_path(void)
{
    ft_tt_test_step("Rung 4b — the BULK UPLOAD PATH: a store through an aperture we mapped, "
                    "read back through TT-KMD's transport");
    upload_and_verify(window_ours_wc, true, L1_SCRATCH_WC, 0x5A5Au, "WC aperture");

    ft_tt_test_step("Rung 4c — the same thing uncached, so the two cache modes are compared "
                    "rather than assumed");
    upload_and_verify(window_ours_uc, false, L1_SCRATCH_UC, 0xA5A5u, "UC aperture");
}

/* ------------------------------------------------------------------------ *
 * Rung 4d — the window walk on silicon, and the re-aim fast path
 * ------------------------------------------------------------------------ */

/**
 * L1 is 1536 KiB, so every L1 transfer fits one 2 MiB window and the walk is
 * never exercised by rung 4b. DRAM is 4 GiB per bank, so a span deliberately
 * placed across a 2 MiB boundary makes the walk move the window mid-transfer —
 * which is the code path that, when wrong, writes to the right endpoint at the
 * wrong offset rather than failing.
 */
static void stage_the_window_walk(void)
{
    ft_tt_udrv_noc_path path;
    ft_tt_noc_coord bank;
    unsigned char *sent = malloc(DRAM_SPAN_BYTES);
    unsigned char *received = malloc(DRAM_SPAN_BYTES);
    uint64_t before;
    uint64_t after;
    int rc;

    ft_tt_test_step("Rung 4d — a transfer that STRADDLES a 2 MiB boundary, so the window really "
                    "moves mid-copy");

    if (!sent || !received) {
        ft_tt_test_check("allocation", false, "");
        goto done;
    }
    if (ft_tt_coord_dram_bank(&session.grid, 0u, FT_TT_COORD_TRANSLATED, &bank) != FT_TT_OK) {
        ft_tt_test_check("a TRANSLATED DRAM bank coordinate resolves", false, "");
        goto done;
    }
    ft_tt_test_check_fmt("a TRANSLATED DRAM bank coordinate resolves", true, "bank 0 at (%u,%u)",
                         bank.x, bank.y);

    ft_tt_test_check_fmt("the span really does straddle a boundary",
                         WINDOW_PAGE(DRAM_SCRATCH_BASE) !=
                             WINDOW_PAGE(DRAM_SCRATCH_BASE + DRAM_SPAN_BYTES - 1u),
                         "0x%llx .. 0x%llx", (unsigned long long)DRAM_SCRATCH_BASE,
                         (unsigned long long)(DRAM_SCRATCH_BASE + DRAM_SPAN_BYTES));

    if (ft_tt_udrv_noc_path_open(&udrv, window_ours_wc, true, &path) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("our aperture maps", false, "errno %d", errno);
        goto done;
    }

    fill_pattern(sent, DRAM_SPAN_BYTES, 0x1234u);
    before = ft_tt_udrv_noc_path_aim_count(&path);
    rc = ft_tt_udrv_noc_transfer(&path, (uint16_t)bank.x, (uint16_t)bank.y, DRAM_SCRATCH_BASE,
                                 sent, DRAM_SPAN_BYTES, false, FT_TT_NOC_WIDTH_BLOCK16);
    after = ft_tt_udrv_noc_path_aim_count(&path);

    ft_tt_test_check_fmt("the straddling write returns OK", rc == FT_TT_UDRV_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("...and it aimed the window TWICE, which is the walk actually walking",
                         after - before == 2u, "%llu aims", (unsigned long long)(after - before));

    memset(received, 0, DRAM_SPAN_BYTES);
    rc = ft_tt_read_dram(&session.runtime.binding.path, bank, DRAM_SCRATCH_BASE, received,
                         DRAM_SPAN_BYTES);
    ft_tt_test_check_fmt("the oracle reads the whole span back byte-exact ACROSS the boundary",
                         rc == FT_TT_OK && memcmp(sent, received, DRAM_SPAN_BYTES) == 0, "rc=%d",
                         rc);

    /* THE RE-AIM FAST PATH, which is the dangerous optimisation in the walk: a
     * transfer to an endpoint the window already points at must cost zero aims,
     * and a transfer to a different endpoint must cost one. */
    before = ft_tt_udrv_noc_path_aim_count(&path);
    (void)ft_tt_udrv_noc_transfer(&path, (uint16_t)bank.x, (uint16_t)bank.y,
                                  DRAM_SCRATCH_BASE + (uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE, sent,
                                  64u, false, FT_TT_NOC_WIDTH_BLOCK16);
    after = ft_tt_udrv_noc_path_aim_count(&path);
    ft_tt_test_check_fmt("a transfer inside the page the window already holds costs ZERO aims",
                         after == before, "%llu", (unsigned long long)(after - before));

    before = after;
    (void)ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, L1_SCRATCH_WC, sent, 64u, false,
                                  FT_TT_NOC_WIDTH_BLOCK16);
    after = ft_tt_udrv_noc_path_aim_count(&path);
    ft_tt_test_check_fmt("...and moving to a different TILE costs exactly one", after - before == 1u,
                         "%llu", (unsigned long long)(after - before));

    ft_tt_test_check_fmt("no aim was ever refused on this path",
                         ft_tt_udrv_noc_path_aim_failures(&path) == 0u, "%llu",
                         (unsigned long long)ft_tt_udrv_noc_path_aim_failures(&path));

    /* Put back what rung 4b left at L1_SCRATCH_WC, so a later arm reading it is
     * not confused by these 64 bytes. Nothing depends on it; it is hygiene. */
    (void)ft_tt_udrv_noc_path_close(&path);

done:
    free(sent);
    free(received);
}

/* ------------------------------------------------------------------------ *
 * Rung 5 — stream_header and static_vc, which the ioctl cannot express
 * ------------------------------------------------------------------------ */

/**
 * The two fields CONFIGURE_TLB has no way to set. tt-kmd folds the ioctl's
 * `static_vc` flag into the register's `use_static_vc` bit and never touches
 * `stream_header` or the 3-bit `static_vc` id at all — so the driver cannot be
 * an oracle for them, they have never been swept, and only our own register
 * writes can reach them.
 *
 * WHAT IS SWEPT HERE, and it is a REGISTER-LEVEL sweep on purpose: every one of
 * the 32 combinations of `stream_header` (2) x `static_vc` (8) x
 * `use_static_vc` (2), written by us and read back through TT-KMD's BAR0
 * mapping. That answers the question that has to be answered first — whether
 * the hardware IMPLEMENTS those bits at all, or reads them back as zero — and
 * it emits no NoC transaction, because the swept window's aperture is never
 * mapped.
 *
 * WHAT IS NOT SWEPT, and why. A FUNCTIONAL arm for a non-zero `static_vc`
 * forces every transaction of that window onto one NoC virtual channel. Nothing
 * in tt-kmd, and nothing this project has read of tt-umd, ever sets a non-zero
 * static VC on a host TLB, so there is no oracle and no source that bounds what
 * a VC reserved for another traffic class does to the fabric. A deadlocked NoC
 * costs a power cycle and a walk to the box. So exactly one functional arm runs:
 * `use_static_vc = 1, static_vc = 0`, which is the configuration tt-kmd's own
 * ioctl produces when a caller passes `static_vc = 1`. Everything above VC 0
 * stays register-level, and that is recorded as unproven rather than skipped
 * quietly.
 */
static void stage_static_vc_sweep(void)
{
    uint32_t seen_words[32][3];
    size_t combination = 0u;
    size_t disagreements = 0u;
    size_t stream_header_held = 0u;
    size_t static_vc_held = 0u;
    size_t duplicates = 0u;
    unsigned use_vc;
    unsigned header;
    unsigned vc;
    size_t left;
    size_t right;

    ft_tt_test_step("Rung 5 — stream_header and the 3-bit static_vc: 32 combinations "
                    "CONFIGURE_TLB cannot express, read back through TT-KMD's mapping");

    for (use_vc = 0u; use_vc <= 1u; use_vc++) {
        for (header = 0u; header <= 1u; header++) {
            for (vc = 0u; vc <= 7u; vc++) {
                ft_tt_udrv_tlb2m_config config;
                ft_tt_udrv_tlb2m_config decoded;
                uint32_t intended[3] = {0u, 0u, 0u};
                uint32_t observed[3] = {0u, 0u, 0u};

                memset(&config, 0, sizeof(config));
                config.addr = WINDOW_PAGE(ARC_CSM_BASE);
                config.x_end = ARC_X;
                config.y_end = ARC_Y;
                config.ordering = 1u;
                config.use_static_vc = (uint8_t)use_vc;
                config.stream_header = (uint8_t)header;
                config.static_vc_id = (uint8_t)vc;

                if (ft_tt_udrv_noc_window_aim(&udrv, window_sweep, &config) != FT_TT_UDRV_OK) {
                    ft_tt_test_check_fmt("our aim self-verifies", false,
                                         "use_vc=%u sh=%u vc=%u", use_vc, header, vc);
                    disagreements++;
                    combination++;
                    continue;
                }

                (void)ft_tt_udrv_tlb2m_encode(&config, intended);
                read_triple_via_driver(window_sweep, observed);
                memcpy(seen_words[combination], observed, sizeof(observed));

                if (memcmp(intended, observed, sizeof(intended)) != 0) {
                    ft_tt_test_check_fmt("the register holds what we encoded", false,
                                         "use_vc=%u sh=%u vc=%u: ours 0x%08x 0x%08x 0x%08x, card "
                                         "0x%08x 0x%08x 0x%08x",
                                         use_vc, header, vc, intended[0], intended[1], intended[2],
                                         observed[0], observed[1], observed[2]);
                    disagreements++;
                } else {
                    /* Decode what the CARD holds, not what we sent, so "the bit
                     * is retained" is a statement about the hardware. */
                    (void)ft_tt_udrv_tlb2m_decode(observed, &decoded);
                    if (decoded.stream_header == (uint8_t)header)
                        stream_header_held++;
                    if (decoded.static_vc_id == (uint8_t)vc)
                        static_vc_held++;
                }
                combination++;
            }
        }
    }

    ft_tt_test_check_fmt("all 32 combinations read back byte-identical through TT-KMD's mapping",
                         disagreements == 0u, "%zu disagreed", disagreements);
    ft_tt_test_check_fmt("THE HARDWARE RETAINS stream_header — it is not a reserved bit that "
                         "reads back zero",
                         stream_header_held == combination, "%zu of %zu", stream_header_held,
                         combination);
    ft_tt_test_check_fmt("THE HARDWARE RETAINS the 3-bit static_vc, all eight values",
                         static_vc_held == combination, "%zu of %zu", static_vc_held, combination);

    /* THE ANTI-VACUITY ARM. If every combination produced the same three words,
     * the three checks above would pass while proving nothing. */
    for (left = 0u; left < combination; left++) {
        for (right = left + 1u; right < combination; right++) {
            if (memcmp(seen_words[left], seen_words[right], sizeof(seen_words[left])) == 0)
                duplicates++;
        }
    }
    ft_tt_test_check_fmt("no two combinations produced the same register words — the sweep is "
                         "actually sweeping",
                         duplicates == 0u, "%zu duplicate pairs of %zu", duplicates,
                         combination * (combination - 1u) / 2u);

    /* THE ONE FUNCTIONAL ARM: use_static_vc = 1, VC 0. See this function's
     * documentation for why it stops there. */
    {
        ft_tt_udrv_noc_path path;
        ft_tt_noc_coord core_a;
        unsigned char *sent = malloc(4096u);
        unsigned char *received = malloc(4096u);

        core_a.x = core_a_x;
        core_a.y = core_a_y;

        if (!sent || !received) {
            ft_tt_test_check("allocation", false, "");
        } else if (ft_tt_udrv_noc_path_open(&udrv, window_ours_uc, false, &path) !=
                   FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt("the static-VC path's aperture maps", false, "errno %d", errno);
        } else {
            ft_tt_udrv_tlb2m_config aim;
            int rc;

            (void)ft_tt_udrv_noc_path_set_aim_template(&path, 0u, 1u, 0u, 1u, 0u, 0u);

            memset(&aim, 0, sizeof(aim));
            aim.addr = WINDOW_PAGE(L1_SCRATCH_UC);
            aim.x_end = (uint8_t)core_a_x;
            aim.y_end = (uint8_t)core_a_y;
            aim.ordering = 1u;
            aim.use_static_vc = 1u;

            if (aim_ours_verified(window_ours_uc, &aim)) {
                fill_pattern(sent, 4096u, 0x777u);
                rc = ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, L1_SCRATCH_UC, sent, 4096u,
                                             false, FT_TT_NOC_WIDTH_BLOCK16);
                ft_tt_test_check_fmt("a write through a window with use_static_vc=1, VC 0 returns "
                                     "OK",
                                     rc == FT_TT_UDRV_OK, "rc=%d", rc);

                memset(received, 0, 4096u);
                rc = ft_tt_read_l1(&session.runtime.binding.path, core_a, L1_SCRATCH_UC, received, 4096u);
                ft_tt_test_check_fmt("...and the oracle reads exactly those bytes back, so a "
                                     "static VC does not silently drop traffic",
                                     rc == FT_TT_OK && memcmp(sent, received, 4096u) == 0, "rc=%d",
                                     rc);
            }
            (void)ft_tt_udrv_noc_path_close(&path);
        }
        free(sent);
        free(received);
    }

    ft_tt_test_check("static_vc 1-7 are proven to be HELD by the register and are NOT proven to "
                     "route — no oracle exists for them and a wrong VC is a fabric hazard",
                     true, "recorded as unproven, deliberately");
}

/* ------------------------------------------------------------------------ *
 * Refusals — the transport must say no before the card is reached
 * ------------------------------------------------------------------------ */

static void stage_refusals(void)
{
    ft_tt_udrv_noc_path path;
    ft_tt_udrv_tlb2m_config config;
    unsigned char scratch[64];

    ft_tt_test_step("Refusals — every one of these must fail BEFORE a transaction is emitted");

    memset(scratch, 0, sizeof(scratch));

    if (ft_tt_udrv_noc_path_open(&udrv, window_ours_uc, false, &path) != FT_TT_UDRV_OK) {
        ft_tt_test_check("a path opens for the refusal arms", false, "");
        return;
    }

    ft_tt_test_check_fmt("a coordinate above 63 is refused, not truncated onto a live tile",
                         ft_tt_udrv_noc_transfer(&path, 64u, 0u, L1_SCRATCH_UC, scratch,
                                                 sizeof(scratch), true,
                                                 FT_TT_NOC_WIDTH_BLOCK16) == FT_TT_UDRV_EINVAL,
                         "x=64");
    ft_tt_test_check("a y above 63 likewise",
                     ft_tt_udrv_noc_transfer(&path, 0u, 64u, L1_SCRATCH_UC, scratch,
                                             sizeof(scratch), true,
                                             FT_TT_NOC_WIDTH_BLOCK16) == FT_TT_UDRV_EINVAL,
                     "");
    ft_tt_test_check("a misaligned address is refused",
                     ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, L1_SCRATCH_UC + 1u,
                                             scratch, sizeof(scratch), true,
                                             FT_TT_NOC_WIDTH_BLOCK16) == FT_TT_UDRV_EINVAL,
                     "");
    ft_tt_test_check("a length that is not a multiple of the width is refused",
                     ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, L1_SCRATCH_UC, scratch, 17u,
                                             true, FT_TT_NOC_WIDTH_BLOCK16) == FT_TT_UDRV_EINVAL,
                     "");
    ft_tt_test_check("a zero-length transfer is a legal no-op",
                     ft_tt_udrv_noc_transfer(&path, core_a_x, core_a_y, L1_SCRATCH_UC, scratch, 0u,
                                             true, FT_TT_NOC_WIDTH_BLOCK16) == FT_TT_UDRV_OK,
                     "");

    memset(&config, 0, sizeof(config));
    config.addr = WINDOW_PAGE(ARC_CSM_BASE);
    config.x_end = 64u; /* does not fit the 6-bit field */
    ft_tt_test_check("an out-of-range aim is refused rather than silently truncated",
                     ft_tt_udrv_noc_window_aim(&udrv, window_sweep, &config) == FT_TT_UDRV_EINVAL,
                     "x_end=64");

    config.x_end = ARC_X;
    config.addr = WINDOW_PAGE(ARC_CSM_BASE) + 4096u; /* not 2 MiB aligned */
    ft_tt_test_check("a misaligned aim address is refused",
                     ft_tt_udrv_noc_window_aim(&udrv, window_sweep, &config) == FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check("an id past the window count is refused",
                     ft_tt_udrv_noc_window_aim(&udrv, FT_TT_UDRV_TLB_2M_WINDOW_COUNT, &config) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    (void)ft_tt_udrv_noc_path_close(&path);
}

/* ------------------------------------------------------------------------ *
 * Release
 * ------------------------------------------------------------------------ */

static void stage_release(void)
{
    ft_tt_udrv_tlb2m_config benign;
    size_t failures = 0u;
    size_t index;

    ft_tt_test_step("Release — re-aim every claimed window somewhere harmless, then hand it back");

    memset(&benign, 0, sizeof(benign));
    benign.addr = WINDOW_PAGE(ARC_CSM_BASE);
    benign.x_end = ARC_X;
    benign.y_end = ARC_Y;
    benign.ordering = 1u;

    for (index = 0u; index < claimed_count; index++) {
        if (configure_window_via_ioctl(claimed[index], &benign) != FT_TT_UDRV_OK)
            failures++;
    }
    ft_tt_test_check_fmt("every window is aimed at the ARC tile, unicast, no multicast bit and no "
                         "static VC left set anywhere",
                         failures == 0u, "%zu of %zu failed", failures, claimed_count);

    failures = 0u;
    for (index = 0u; index < claimed_count; index++) {
        if (free_window(claimed[index]) != FT_TT_UDRV_OK)
            failures++;
    }
    ft_tt_test_check_fmt("every window is released", failures == 0u, "%zu of %zu failed", failures,
                         claimed_count);
    claimed_count = 0u;

    if (driver_registers) {
        (void)munmap((void *)(uintptr_t)driver_registers, FT_TT_UDRV_TLB_REGS_LEN);
        driver_registers = NULL;
    }
    if (udrv_open) {
        (void)ft_tt_udrv_noc_device_close(&udrv);
        udrv_open = false;
    }
    if (chardev >= 0) {
        (void)close(chardev);
        chardev = -1;
    }
    if (session_open) {
        ft_tt_hardware_session_close(&session);
        session_open = false;
    }
    ft_tt_test_check("mappings unmapped, cores released, chardev closed", true, "");
}

/* ------------------------------------------------------------------------ */

static void usage(const char *program)
{
    (void)fprintf(stderr,
                  "usage: %s [read | write]\n"
                  "  read   rungs 1-3: map, compare both readers, and read through a window we\n"
                  "         aimed. WRITES NOTHING TO THE CARD beyond the window aim itself.\n"
                  "  write  the above, plus the write-combined store arms, the window walk and\n"
                  "         the stream_header/static_vc sweep.\n"
                  "Default is `read`.\n",
                  program);
}

int main(int argc, char **argv)
{
    bool writing = false;

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

    if (!stage_open_the_card(writing))
        return ft_tt_test_finish();
    if (!stage_open_the_session()) {
        stage_release();
        return ft_tt_test_finish();
    }
    if (!stage_claim_windows()) {
        stage_release();
        return ft_tt_test_finish();
    }

    if (!writing) {
        ft_tt_test_step("Rungs 3-5 SKIPPED — aiming a window is a WRITE, so the read-only run "
                        "stops here. Run with `write`.");
        stage_release();
        return ft_tt_test_finish();
    }

    stage_read_agreement();
    stage_wc_store_reaches_the_card();
    stage_the_upload_path();
    stage_the_window_walk();
    stage_static_vc_sweep();
    stage_refusals();
    stage_release();
    return ft_tt_test_finish();
}
