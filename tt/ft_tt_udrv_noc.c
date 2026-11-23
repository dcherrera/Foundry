/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_noc.c
 * @brief Implementation of the userspace driver's NoC transport: aim a 2 MiB
 *   window with our own register writes, verify the aim before anything
 *   dereferences the aperture, and walk that window across a transfer of any
 *   length.
 *
 *   The copy itself is not here. It is `ft_tt_noc_copy.h`, shared with the
 *   tt-kmd transport, so there is exactly one PIO copy shape in the backend.
 *
 * ---------------------------------------------------------------------------
 * WHY THE WRITE PATH FENCES AND ft_tt_noc.c DOES NOT
 * ---------------------------------------------------------------------------
 * A write-combined store sits in the core's WC buffers until something retires
 * it. The tt-kmd transport has only ever had the COMPILER barrier at the end of
 * a slice, which orders the instruction stream and nothing else; it gets away
 * with it because every consumer of an upload so far has been a later NoC
 * transaction through the same window, and the buffers drain long before one
 * matters. That is a property of the callers, not of the code. This path is
 * read back deliberately, immediately, and through a DIFFERENT mapping, so it
 * calls `ft_tt_noc_copy_store_fence()` — an `sfence` — after the last store of
 * every write slice.
 *
 * AND THE MEASUREMENT SAYS IT IS NOT NEEDED HERE. Removing every fence, from
 * this file and from the on-card test, left the whole suite green on the card host
 * (2026-08-17): the WC stores landed anyway. That is a fact about one host's
 * buffers and access pattern, not a guarantee, and the target is a machine
 * nobody here has — so the fence stays, correctly described as cheap insurance
 * rather than as a demonstrated requirement.
 *
 * @status On silicon 2026-08-17 — see the header.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_udrv_noc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ft_tt_udrv.h"

/** How much of an aim is read back. FULL is four non-posted reads and is the
 *  default; the other two exist to PRICE the check, not to ship. */
typedef enum udrv_aim_verify {
    UDRV_AIM_VERIFY_FULL = 0, /**< three config words plus the strided register */
    UDRV_AIM_VERIFY_WORD = 1, /**< word 1 alone — see the note in the aim */
    UDRV_AIM_VERIFY_NONE = 2  /**< nothing; the ceiling, and unsafe */
} udrv_aim_verify;

/**
 * Parse `FT_TT_UDRV_AIM_VERIFY` once.
 *
 * WORD reads register word 1 and only word 1, which is the word that carries
 * `x_end` (bit 43) and `y_end` (bit 49) — so it is the cheapest readback that
 * still checks the ENDPOINT. It gives up the strided-multicast register, and a
 * stale strided pattern on a window below index 32 is a MULTICAST: this
 * project's most reliable way to wedge the NoC.
 *
 * ---------------------------------------------------------------------------
 * AND IT GIVES UP THE WHOLE ADDRESS, NOT "THE LOW ADDRESS BITS"
 * ---------------------------------------------------------------------------
 * READ THE BIT LAYOUT, NOT THE ADJECTIVE. The address field is
 * `addr >> FT_TT_UDRV_TLB_2M_SHIFT` at bit 0, 43 bits wide, so register **word
 * 0 IS address bits 21..52** and word 1 carries only address bits **53..63** —
 * which are ZERO for every endpoint on this part, DRAM included (32 GiB is
 * 2^35, and the highest L1 base the dispatch path aims at is 0x0ffa00000, well
 * inside 2^36). **So the eleven address bits `word` verifies are eleven zeros,
 * and not one significant address bit is read back.**
 *
 * The consequence is precise and it is not a hang: a store that lands wrong in
 * word 0 aims the window at a DIFFERENT 2 MiB page OF THE RIGHT CORE, `word`
 * sees the right `x_end`/`y_end` and agrees, and the descriptor is then written
 * into whatever is 2 MiB away. On the register bank the two bases in play are
 * L1 at 0 and the run-control block at 0x0ffa00000, so the pages either side of
 * a live one hold a resident kernel's code and its run control. **The failure
 * mode of `word` is silent corruption of the thing that is running, not an
 * error and not a stall.**
 *
 * `FT_TT_UDRV_AIM_RED=aimlow` is the arm that shows this rather than arguing
 * it: it perturbs word 0 after the stores, so `full` must refuse and `word`
 * must not notice. See red_aim_low() below for why it takes two keys.
 */
static udrv_aim_verify udrv_aim_verify_selected(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("FT_TT_UDRV_AIM_VERIFY");

        cached = (int)UDRV_AIM_VERIFY_FULL;
        if (value && *value) {
            if (strcmp(value, "full") == 0)
                cached = (int)UDRV_AIM_VERIFY_FULL;
            else if (strcmp(value, "word") == 0)
                cached = (int)UDRV_AIM_VERIFY_WORD;
            else if (strcmp(value, "none") == 0)
                cached = (int)UDRV_AIM_VERIFY_NONE;
            else
                fprintf(stderr,
                        "ft_tt_udrv_noc: FT_TT_UDRV_AIM_VERIFY=%s is not full|word|none — "
                        "using full\n",
                        value);
        }
    }
    return (udrv_aim_verify)cached;
}

/** The second key `aimlow` needs. See red_aim_low(). */
#define UDRV_AIM_RED_CONFIRM "i-know-this-aims-at-the-wrong-page"

/**
 * Whether the word-0 red arm is armed, and it takes two keys.
 *
 * ---------------------------------------------------------------------------
 * THIS IS THE ARM THAT PRICES `FT_TT_UDRV_AIM_VERIFY=word`'s SAFETY, AND IT
 * WRITES TO THE WRONG PAGE ON PURPOSE
 * ---------------------------------------------------------------------------
 * A `word` verification reads back no significant address bit, so the argument
 * that it is weaker cannot be settled by reading it — it has to be shown. This
 * arm perturbs register word 0 AFTER the three config stores and before the
 * readback, which moves the window by exactly one 2 MiB page and leaves
 * `x_end`/`y_end` alone. Then:
 *
 *   full   the readback disagrees and the aim REFUSES with EIO. Nothing is
 *          dereferenced, which is the whole design of a non-posted verify.
 *   word   the readback agrees, because the bits it reads did not move, and
 *          the transfer lands one page away. **That is the measurement.**
 *
 * **DRAM ENDPOINTS ONLY, THROUGH `tests/test_udrv_card_bank.c`.** One page off a
 * DRAM base is another part of the same 4 GiB bank and is inert; one page off an
 * L1 base is off the end of a 1.5 MiB L1, and a NoC transaction to an address a
 * tile does not decode is how this program lost a box to a power cycle on
 * 2026-08-26.
 *
 * ---------------------------------------------------------------------------
 * AND NO ENVIRONMENT ALONE CAN ARM IT, WHICH IS A MEASURED CORRECTION
 * ---------------------------------------------------------------------------
 * The first version armed itself from the environment on first use, and the two
 * arms that needed it both died BEFORE their own first check — because the first
 * aims of a process are the card BRING-UP's, so `full` refused the telemetry read
 * and `word` accepted it and read the wrong page. **An arm that fires during
 * bring-up is an arm that never reaches the thing under test.** So the keys are
 * PERMISSION and `ft_tt_udrv_noc_aim_red_arm()` is TIMING: a caller opens the
 * card honestly, arms, runs one transfer, and disarms. That also means an env
 * line on a model run can no longer arm anything, which is the safer default of
 * the two.
 */
static bool aim_red_low_on;

static bool red_aim_low_armed(void)
{
    return aim_red_low_on;
}

bool ft_tt_udrv_noc_aim_red_arm(bool on)
{
    const char *value = getenv("FT_TT_UDRV_AIM_RED");
    const char *confirm = getenv("FT_TT_UDRV_AIM_RED_CONFIRM");

    if (!on) {
        if (aim_red_low_on)
            fprintf(stderr, "ft_tt_udrv_noc: FT_TT_UDRV_AIM_RED=aimlow DISARMED\n");
        aim_red_low_on = false;
        return false;
    }
    if (!value || strcmp(value, "aimlow") != 0) {
        fprintf(stderr,
                "ft_tt_udrv_noc: the word-0 arm was requested and FT_TT_UDRV_AIM_RED does not "
                "say aimlow — NOT ARMED\n");
        return false;
    }
    if (!confirm || strcmp(confirm, UDRV_AIM_RED_CONFIRM) != 0) {
        fprintf(stderr,
                "ft_tt_udrv_noc: FT_TT_UDRV_AIM_RED=aimlow NOT ARMED — it also needs "
                "FT_TT_UDRV_AIM_RED_CONFIRM=%s. It aims every window ONE 2 MiB PAGE past "
                "where the caller asked, which off an L1 base is an address no tile decodes\n",
                UDRV_AIM_RED_CONFIRM);
        return false;
    }
    fprintf(stderr,
            "ft_tt_udrv_noc: FT_TT_UDRV_AIM_RED=aimlow IS ARMED. Every aim from here on is "
            "perturbed in register word 0 by one 2 MiB page. `full` MUST refuse it and `word` "
            "MUST NOT — that disagreement is the measurement\n");
    aim_red_low_on = true;
    return true;
}

/** Seconds on a monotonic clock. Used only to price the AIM, which is a per-descriptor cost on
 *  the dispatch path and so has to be a number rather than an inference. */
static double udrv_now_seconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/** Word index of window `id`'s register triple within the mapped register file. */
static size_t register_word_index(uint32_t id)
{
    return (size_t)id * (FT_TT_UDRV_TLB_REG_SIZE / 4u);
}

/** Word index of window `id`'s strided-multicast register, for id < 32. */
static size_t strided_word_index(uint32_t id)
{
    return (size_t)(FT_TT_UDRV_TLB_STRIDED_REGS_OFFSET + id * FT_TT_UDRV_TLB_STRIDED_REG_SIZE) / 4u;
}

/* ------------------------------------------------------------------------- */
/* The device                                                                */
/* ------------------------------------------------------------------------- */

int ft_tt_udrv_noc_device_open(const ft_tt_udrv_pci_device *pci, bool writable,
                               ft_tt_udrv_noc_device *out_device)
{
    void *mapping = NULL;
    int rc;

    if (!pci || !out_device)
        return FT_TT_UDRV_EINVAL;

    memset(out_device, 0, sizeof(*out_device));

    /* Uncached, through `resourceN`. Registers are not the write-combined path
     * and never will be: WC is allowed to merge and reorder, and the aim is
     * three words whose order tt-kmd itself is careful about. */
    rc = ft_tt_udrv_pci_map_bar(pci, 0u, false, writable, FT_TT_UDRV_TLB_REGS_START,
                                FT_TT_UDRV_TLB_REGS_LEN, &mapping);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    out_device->pci = *pci;
    out_device->registers = mapping;
    out_device->writable = writable;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_device_close(ft_tt_udrv_noc_device *device)
{
    int rc = FT_TT_UDRV_OK;

    if (!device)
        return FT_TT_UDRV_EINVAL;
    if (device->registers)
        rc = ft_tt_udrv_pci_unmap_bar((void *)(uintptr_t)device->registers,
                                      FT_TT_UDRV_TLB_REGS_LEN);

    memset(device, 0, sizeof(*device));
    return rc;
}

int ft_tt_udrv_noc_window_read(const ft_tt_udrv_noc_device *device, uint32_t id,
                               uint32_t out_words[3])
{
    size_t base;

    if (!device || !device->registers || !out_words)
        return FT_TT_UDRV_EINVAL;
    if (id >= FT_TT_UDRV_TLB_2M_WINDOW_COUNT)
        return FT_TT_UDRV_EINVAL;

    base = register_word_index(id);
    out_words[0] = device->registers[base + 0u];
    out_words[1] = device->registers[base + 1u];
    out_words[2] = device->registers[base + 2u];
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_window_aim_timed(ft_tt_udrv_noc_device *device, uint32_t id,
                                    const ft_tt_udrv_tlb2m_config *config,
                                    double *out_verify_seconds)
{
    uint32_t intended[3] = {0u, 0u, 0u};
    uint32_t observed[3] = {0u, 0u, 0u};
    volatile uint32_t *registers;
    double verify_from;
    size_t base;
    int rc;

    if (out_verify_seconds)
        *out_verify_seconds = 0.0;

    if (!device || !device->registers || !config)
        return FT_TT_UDRV_EINVAL;
    if (id >= FT_TT_UDRV_TLB_2M_WINDOW_COUNT)
        return FT_TT_UDRV_EINVAL;
    if (!device->writable)
        return FT_TT_UDRV_ENOTSUP;

    /* REFUSE a config that does not fit rather than truncating it. The encoder
     * truncates because it must match the kernel's bitfield assignment bit for
     * bit; a caller aiming a window has no such obligation, and a coordinate of
     * 64 folded onto 0 is a different, live endpoint. */
    rc = ft_tt_udrv_tlb2m_config_check(config);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = ft_tt_udrv_tlb2m_encode(config, intended);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    registers = device->registers;
    base = register_word_index(id);

    /* tt-kmd's order: low, mid, high, then the strided register cleared. A
     * freshly claimed window is NOT blank and may still hold a strided
     * multicast pattern, so the zeroing write is part of aiming, not an
     * optimisation (ft_tt_udrv_tlb_encode.h). */
    registers[base + 0u] = intended[0];
    registers[base + 1u] = intended[1];
    registers[base + 2u] = intended[2];
    if (id < FT_TT_UDRV_TLB_STRIDED_COUNT)
        registers[strided_word_index(id)] = 0u;

    /* THE WORD-0 RED ARM. Moves the window one 2 MiB page and leaves the
     * endpoint alone, so `full` refuses and `word` cannot. `intended` is NOT
     * updated: what is under test is whether the readback catches a register
     * that disagrees with what the caller asked for. */
    if (red_aim_low_armed())
        registers[base + 0u] = intended[0] ^ 1u;

    /* THE AIM IS NOT DONE UNTIL IT HAS BEEN READ BACK. Three MMIO stores cannot
     * report a failure the way an ioctl can, and the failure mode is a wrong
     * ADDRESS rather than an error.
     *
     * TIMED SEPARATELY FROM THE STORES BECAUSE THEY ARE DIFFERENT KINDS OF ACCESS. The four
     * stores above are POSTED and retire at store-buffer speed; these are NON-POSTED reads and
     * each is a PCIe round trip. On the dispatch path a window is re-aimed once per descriptor,
     * so whichever half dominates is a per-descriptor cost — and which half it is decides
     * whether the fix is fewer readbacks or more windows. */
    verify_from = udrv_now_seconds();
    switch (udrv_aim_verify_selected()) {
    case UDRV_AIM_VERIFY_NONE:
        rc = FT_TT_UDRV_OK;
        break;
    case UDRV_AIM_VERIFY_WORD:
        observed[1] = registers[base + 1u];
        rc = (observed[1] != intended[1]) ? FT_TT_UDRV_EIO : FT_TT_UDRV_OK;
        break;
    case UDRV_AIM_VERIFY_FULL:
    default:
        observed[0] = registers[base + 0u];
        observed[1] = registers[base + 1u];
        observed[2] = registers[base + 2u];
        rc = (memcmp(intended, observed, sizeof(intended)) != 0) ? FT_TT_UDRV_EIO
                                                                 : FT_TT_UDRV_OK;
        if (rc == FT_TT_UDRV_OK && id < FT_TT_UDRV_TLB_STRIDED_COUNT &&
            registers[strided_word_index(id)] != 0u)
            rc = FT_TT_UDRV_EIO;
        break;
    }
    if (out_verify_seconds)
        *out_verify_seconds = udrv_now_seconds() - verify_from;

    return rc;
}

int ft_tt_udrv_noc_window_aim(ft_tt_udrv_noc_device *device, uint32_t id,
                              const ft_tt_udrv_tlb2m_config *config)
{
    return ft_tt_udrv_noc_window_aim_timed(device, id, config, NULL);
}

/* ------------------------------------------------------------------------- */
/* Path lifecycle                                                            */
/* ------------------------------------------------------------------------- */

int ft_tt_udrv_noc_path_open(ft_tt_udrv_noc_device *device, uint32_t id, bool write_combine,
                             ft_tt_udrv_noc_path *out_path)
{
    uint64_t aperture_offset = 0u;
    void *mapping = NULL;
    int rc;

    if (!device || !device->registers || !out_path)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_tlb2m_aperture_offset(id, &aperture_offset);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    memset(out_path, 0, sizeof(*out_path));

    rc = ft_tt_udrv_pci_map_bar(&device->pci, 0u, write_combine, device->writable,
                                aperture_offset, (size_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE, &mapping);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    out_path->device = device;
    out_path->window_id = id;
    out_path->aperture = mapping;
    out_path->window_bytes = (size_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE;
    out_path->write_combined = write_combine;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_path_close(ft_tt_udrv_noc_path *path)
{
    int rc = FT_TT_UDRV_OK;

    if (!path)
        return FT_TT_UDRV_EINVAL;
    if (path->aperture)
        rc = ft_tt_udrv_pci_unmap_bar(path->aperture, path->window_bytes);

    memset(path, 0, sizeof(*path));
    return rc;
}

int ft_tt_udrv_noc_path_set_aim_template(ft_tt_udrv_noc_path *path, uint8_t noc, uint8_t ordering,
                                         uint8_t linked, uint8_t use_static_vc,
                                         uint8_t static_vc_id, uint8_t stream_header)
{
    ft_tt_udrv_tlb2m_config candidate;

    if (!path)
        return FT_TT_UDRV_EINVAL;

    memset(&candidate, 0, sizeof(candidate));
    candidate.noc = noc;
    candidate.ordering = ordering;
    candidate.linked = linked;
    candidate.use_static_vc = use_static_vc;
    candidate.static_vc_id = static_vc_id;
    candidate.stream_header = stream_header;

    if (ft_tt_udrv_tlb2m_config_check(&candidate) != FT_TT_UDRV_OK)
        return FT_TT_UDRV_EINVAL;

    path->aim_template = candidate;
    path->aimed = false;
    return FT_TT_UDRV_OK;
}

uint64_t ft_tt_udrv_noc_path_aim_count(const ft_tt_udrv_noc_path *path)
{
    return path ? path->aim_count : 0u;
}

uint64_t ft_tt_udrv_noc_path_aim_failures(const ft_tt_udrv_noc_path *path)
{
    return path ? path->aim_failures : 0u;
}

double ft_tt_udrv_noc_path_aim_seconds(const ft_tt_udrv_noc_path *path)
{
    return path ? path->aim_seconds : 0.0;
}

double ft_tt_udrv_noc_path_aim_verify_seconds(const ft_tt_udrv_noc_path *path)
{
    return path ? path->aim_verify_seconds : 0.0;
}

/* ------------------------------------------------------------------------- */
/* The window walk                                                           */
/* ------------------------------------------------------------------------- */

/**
 * Point the path's window at `base` on (x, y) unless it is already there.
 *
 * The skip is safe only because the path is the sole thing that aims its
 * window. It matters: a run of small transfers to one core would otherwise pay
 * a re-aim per call for a window that never moved.
 */
static int aim_if_needed(ft_tt_udrv_noc_path *path, uint16_t x, uint16_t y, uint64_t base)
{
    ft_tt_udrv_tlb2m_config wanted = path->aim_template;
    double verify_seconds = 0.0;
    double aim_from;
    int rc;

    wanted.addr = base;
    wanted.x_end = (uint8_t)x;
    wanted.y_end = (uint8_t)y;

    if (path->aimed && ft_tt_udrv_tlb2m_config_equal(&path->aim, &wanted))
        return FT_TT_UDRV_OK;

    path->aimed = false;
    aim_from = udrv_now_seconds();
    rc = ft_tt_udrv_noc_window_aim_timed(path->device, path->window_id, &wanted,
                                         &verify_seconds);
    path->aim_seconds += udrv_now_seconds() - aim_from;
    path->aim_verify_seconds += verify_seconds;
    if (rc != FT_TT_UDRV_OK) {
        if (rc == FT_TT_UDRV_EIO)
            path->aim_failures++;
        return rc;
    }

    path->aim = wanted;
    path->aimed = true;
    path->aim_count++;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_transfer(ft_tt_udrv_noc_path *path, uint16_t x, uint16_t y, uint64_t address,
                            void *host, size_t bytes, bool is_read,
                            ft_tt_noc_transfer_width width)
{
    unsigned char *host_bytes = (unsigned char *)host;
    uint32_t unit;

    if (!path || !path->device || !path->aperture)
        return FT_TT_UDRV_EINVAL;
    if (!host && bytes != 0u)
        return FT_TT_UDRV_EINVAL;
    if (!is_read && !path->device->writable)
        return FT_TT_UDRV_ENOTSUP;

    unit = ft_tt_noc_transfer_width_bytes(width);
    if (unit == 0u)
        return FT_TT_UDRV_EINVAL;
    if ((address % unit) != 0u || (bytes % unit) != 0u)
        return FT_TT_UDRV_EINVAL;

    /* x_end/y_end are 6-bit fields. A coordinate that does not fit them would
     * be silently truncated into a different, LIVE endpoint — refuse instead. */
    if (x > 63u || y > 63u)
        return FT_TT_UDRV_EINVAL;

    while (bytes > 0u) {
        ft_tt_noc_window_slice slice;
        volatile unsigned char *aperture;
        int rc;

        if (!ft_tt_noc_window_slice_compute(address, bytes, path->window_bytes, &slice))
            return FT_TT_UDRV_EINVAL;

        rc = aim_if_needed(path, x, y, slice.window_base);
        if (rc != FT_TT_UDRV_OK)
            return rc;

        aperture = (volatile unsigned char *)path->aperture;
        ft_tt_noc_copy_slice(aperture + slice.offset_in_window, host_bytes, slice.bytes, is_read,
                             width);

        /* Retire the write-combining buffers before the caller — or the next
         * slice's re-aim — can observe anything. See this file's header note. */
        if (!is_read)
            ft_tt_noc_copy_store_fence();

        address += slice.bytes;
        host_bytes += slice.bytes;
        bytes -= slice.bytes;

        if (is_read)
            path->bytes_read += slice.bytes;
        else
            path->bytes_written += slice.bytes;
    }
    return FT_TT_UDRV_OK;
}

/* ------------------------------------------------------------------------- */
/* ft_tt_noc_access-shaped adapters                                          */
/* ------------------------------------------------------------------------- */

int ft_tt_udrv_noc_access_write(void *user, uint16_t x, uint16_t y, uint64_t address,
                                const void *src, size_t bytes)
{
    /* Const is dropped so one walk serves both directions; with is_read false
     * nothing is ever written through this pointer. */
    return ft_tt_udrv_noc_transfer((ft_tt_udrv_noc_path *)user, x, y, address,
                                   (void *)(uintptr_t)src, bytes, false,
                                   FT_TT_NOC_WIDTH_WORD32);
}

int ft_tt_udrv_noc_access_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                               size_t bytes)
{
    return ft_tt_udrv_noc_transfer((ft_tt_udrv_noc_path *)user, x, y, address, dst, bytes, true,
                                   FT_TT_NOC_WIDTH_WORD32);
}
