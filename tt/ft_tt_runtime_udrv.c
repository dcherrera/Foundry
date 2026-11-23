/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_runtime_udrv.c
 * @brief Implementation of the userspace-driver half of the runtime (see
 *   ft_tt_runtime_udrv.h).
 *
 * @status ON SILICON 2026-08-17 with `tenstorrent` unloaded: a real
 *   `pantheon serve` answered a chat on the card over this path. Numbers and
 *   the priced 6.6% are in ft_tt_runtime_udrv.h's @status.
 * @issues None
 * @todo None
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_runtime_udrv.h"

#include <stdlib.h>
#include <string.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_device.h"
#include "ft_tt_noc.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_arc_probe.h"
#include "ft_tt_udrv_claim.h"
#include "ft_tt_udrv_noc.h"
#include "ft_tt_udrv_noc_bank.h"
#include "ft_tt_udrv_pci_sysfs.h"
#include "ft_tt_udrv_reset.h"

/** How many Tenstorrent devices this will enumerate before giving up looking
 *  for `device_index`. A host with more cards than this is a refusal, not a
 *  silently truncated scan. */
#define RUNTIME_UDRV_MAX_DEVICES 8u

/** Two KINDS of window: one UC (run control + the ARC) and one WC (bulk) — the
 *  same split the tt-kmd path allocates, for the same two reasons
 *  (ft_tt_runtime.h's "WHY TWO WINDOWS").
 *
 *  EACH KIND IS A BANK OF WINDOWS NOW, NOT ONE. A single window is a one-entry
 *  cache over an access stream that rotates across the resident cores, so it
 *  missed on nearly every descriptor and each miss cost an 8.92 us aim —
 *  4,333 a token, 25.1% of the decode wall. `FT_TT_UDRV_WINDOWS` sizes both
 *  banks; 1 reproduces the single-window transport exactly. */
struct ft_tt_runtime_udrv {
    ft_tt_udrv_pci_device pci;
    const ft_tt_arch *arch;

    ft_tt_udrv_noc_device device;
    bool device_open;

    ft_tt_udrv_claim_namespace windows;

    ft_tt_udrv_noc_bank control; /**< UC — registers, mailbox, images, ARC */
    bool control_open;
    ft_tt_udrv_noc_bank bulk; /**< WC — DRAM and L1 bulk data */
    bool bulk_open;
    /** WC — a THIRD bank, owned by ONE other thread (`ft_tt_runtime_udrv_reader_access()`):
     *  PHASE W, the prefill's alpha/beta reads off the issuing thread. Windows are per-bank
     *  and the device's register mapping is read-only after open, so two threads each on
     *  their own bank never touch the same mutable state. */
    ft_tt_udrv_noc_bank reader[FT_TT_UDRV_READER_BANKS_MAX];
    uint32_t reader_banks;

    ft_tt_udrv_arc_noc_context arc;
    ft_tt_udrv_arc_transport arc_transport;
};

/* ================================================================
 * Status translation
 * ================================================================ */

/**
 * Map an FT_TT_UDRV_* code onto the FT_TT_* code the backend above the seam
 * speaks.
 *
 * The two enums share their first six values by construction (ft_tt_udrv.h),
 * but they DIVERGE at -6 and -7: FT_TT_EPERM/FT_TT_EUNAVAIL against
 * FT_TT_UDRV_EAGAIN/FT_TT_UDRV_ENOTSUP. A memcpy of the number would therefore
 * report "blocked by the Phase 1a NoC fence" for "the ring is full", which is
 * the sort of wrong error message that costs an afternoon. Written out.
 */
static int udrv_status(int udrv_rc)
{
    switch (udrv_rc) {
    case FT_TT_UDRV_OK:
        return FT_TT_OK;
    case FT_TT_UDRV_ENOMEM:
        return FT_TT_ENOMEM;
    case FT_TT_UDRV_EINVAL:
        return FT_TT_EINVAL;
    case FT_TT_UDRV_EIO:
        return FT_TT_EIO;
    case FT_TT_UDRV_ENODEV:
        return FT_TT_ENODEV;
    case FT_TT_UDRV_EAGAIN:
    case FT_TT_UDRV_ENOTSUP:
    case FT_TT_UDRV_ERROR:
    default:
        return FT_TT_ERROR;
    }
}

/* ================================================================
 * The three accessors
 * ================================================================ */

/**
 * The run-control vtable. `ft_tt_udrv_noc_access_write/read` already carry the
 * `ft_tt_noc_access` signature at WORD32 width, so the whole of run control —
 * soft reset, mailbox, go messages, ELF images — runs over them unchanged.
 * Their `user` is the UC path.
 *
 * Nothing is wrapped here on purpose: a wrapper would be a place for the two
 * transports to drift.
 */

/** BLOCK16 over the WC path, with the memory space's own bounds and alignment.
 *
 *  The check is `ft_tt_noc_check_transfer()` — the SAME function ft_tt_noc.c's
 *  ft_tt_write_dram() calls — so a register address is refused here for exactly
 *  the reason and with exactly the message it is refused on the tt-kmd path.
 *  That structural separation (ft_tt_runtime.h) is a correctness property, not
 *  a convention, and it must not be re-derived per transport. */
static int bulk_transfer(void *user, ft_tt_memory_space space, bool is_read, uint16_t x,
                         uint16_t y, uint64_t address, void *host, size_t bytes)
{
    ft_tt_udrv_noc_bank *bank = (ft_tt_udrv_noc_bank *)user;
    int rc;

    if (!bank || !host)
        return FT_TT_EINVAL;
    rc = ft_tt_noc_check_transfer(space, is_read, address, bytes);
    if (rc != FT_TT_OK)
        return rc;
    /* The store fence after the last slice lives inside ft_tt_udrv_noc_transfer()
     * — a WC store may sit in the write-combining buffer, and correctness must
     * not depend on an unrelated module's later UC access draining it. */
    return udrv_status(ft_tt_udrv_noc_bank_transfer(bank, x, y, address, host, bytes, is_read,
                                                    FT_TT_NOC_WIDTH_BLOCK16));
}

static int bulk_dram_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                           size_t bytes)
{
    /* Const is dropped so one walk serves both directions; with is_read false
     * nothing is ever written through this pointer. */
    return bulk_transfer(user, FT_TT_MEM_SPACE_DRAM, false, x, y, addr,
                         (void *)(uintptr_t)src, bytes);
}

static int bulk_dram_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                          size_t bytes)
{
    return bulk_transfer(user, FT_TT_MEM_SPACE_DRAM, true, x, y, addr, dst, bytes);
}

static int bulk_l1_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                         size_t bytes)
{
    return bulk_transfer(user, FT_TT_MEM_SPACE_L1, false, x, y, addr, (void *)(uintptr_t)src,
                         bytes);
}

static int bulk_l1_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst, size_t bytes)
{
    return bulk_transfer(user, FT_TT_MEM_SPACE_L1, true, x, y, addr, dst, bytes);
}

/* ================================================================
 * Discovery
 * ================================================================ */

/** Find the `device_index`-th Tenstorrent card in sysfs and select its
 *  architecture table from the PCI ids alone. Any Tenstorrent device id is
 *  enumerated and the arch table decides — matching one hardcoded id here would
 *  hide a part we can name but do not support behind "no device". */
static int find_card(int device_index, ft_tt_runtime_udrv *udrv)
{
    ft_tt_udrv_pci_device found[RUNTIME_UDRV_MAX_DEVICES];
    size_t count = 0;
    int rc;

    if (device_index < 0) {
        ft_tt_log_error("tt runtime udrv: device index %d is negative", device_index);
        return FT_TT_EINVAL;
    }
    rc = ft_tt_udrv_pci_find_devices(NULL, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT, 0xFFFFu, found,
                                     RUNTIME_UDRV_MAX_DEVICES, &count);
    if (rc != FT_TT_UDRV_OK) {
        ft_tt_log_error("tt runtime udrv: cannot read %s", FT_TT_UDRV_PCI_DEFAULT_DEVICES_DIR);
        return udrv_status(rc);
    }
    if (count == 0u) {
        ft_tt_log_error("tt runtime udrv: no Tenstorrent device in %s — REFUSING. This path "
                        "does not fall back to anything",
                        FT_TT_UDRV_PCI_DEFAULT_DEVICES_DIR);
        return FT_TT_ENODEV;
    }
    if ((size_t)device_index >= count || (size_t)device_index >= RUNTIME_UDRV_MAX_DEVICES) {
        ft_tt_log_error("tt runtime udrv: asked for device %d and sysfs publishes %zu",
                        device_index, count);
        return FT_TT_ENODEV;
    }

    udrv->pci = found[device_index];
    udrv->arch = ft_tt_arch_select_by_pci_ids(udrv->pci.vendor, udrv->pci.device);
    if (!udrv->arch)
        return FT_TT_ENODEV;
    return FT_TT_OK;
}

/**
 * Open one bank of windows, preferring HIGH indices.
 *
 * `ALLOCATE_TLB` allocates from the bottom of the range and our flock cannot
 * see its bitmap, so while tt-kmd is loaded a bottom-up claim here and a
 * bottom-up allocation there converge on the same window and neither party is
 * told. Walking down from the top does not make that safe — nothing can, short
 * of the module being gone — but it removes the collision that would otherwise
 * happen on the FIRST window every time. The window the driver keeps for itself
 * (`arch->tlb.kernel_window_index`, 201) is skipped outright: claiming it out
 * from under a loaded tt-kmd corrupts the driver's own telemetry reads.
 */
static int open_window_bank(ft_tt_runtime_udrv *udrv, bool write_combine, const char *name,
                            uint32_t windows, ft_tt_udrv_noc_bank *out_bank)
{
    int rc = ft_tt_udrv_noc_bank_open(&udrv->device, &udrv->windows,
                                      udrv->arch->tlb.kernel_window_index, write_combine,
                                      windows, name, out_bank);

    if (rc == FT_TT_UDRV_OK)
        return FT_TT_OK;
    /* EAGAIN is contention — somebody holds every window, which is the normal
     * reason to fail. ANYTHING ELSE is the claim FILE not being usable, and
     * reporting that as contention is how a permission problem in /tmp reads as
     * "the card is busy". It cost this gate a round: a root run had left the
     * tree 0755 root-owned and every non-root job then printed "all 202 windows
     * are claimed by a live process" on a completely idle card. */
    if (rc == FT_TT_UDRV_EAGAIN)
        ft_tt_log_error("tt runtime udrv: every one of the %u windows is claimed by a live "
                        "process",
                        udrv->windows.count);
    else
        ft_tt_log_error("tt runtime udrv: the %s window bank could not be opened (rc=%d) — this "
                        "is the flock file under /tmp, not the card. Check the ownership of "
                        "/tmp/ft_tt_udrv.d: a root run that created it under a umask leaves it "
                        "unusable by every later non-root job",
                        name, rc);
    return FT_TT_ERROR;
}

/* ================================================================
 * Bring-up
 * ================================================================ */

/**
 * MRRS — step 1 of `blackhole_init_hardware()`, and the only step of it that is
 * not an ARC message.
 *
 * It reads 4096 on this host whether or not the module is loaded, because tt-kmd
 * once set it and `pci_disable_device()` does not clear DEVCTL. **THE TARGET
 * BOARD HAS NOBODY TO SET IT.** So the bring-up sets it, exactly as the vendor
 * probe does.
 *
 * BEST EFFORT ON PURPOSE, and this is a decision rather than laziness: a
 * config-space write is privileged, so a NON-ROOT run — which is the whole point
 * of the udev rule that lets an ordinary user drive the BARs — cannot do it.
 * Refusing the card over it would make the packaging goal and the probe-parity
 * goal contradict each other. It is loud instead: the failure names itself, and
 * the value it is leaving behind is one that was READ, not assumed.
 */
static void set_max_read_request_size(ft_tt_runtime_udrv *udrv)
{
    ft_tt_udrv_max_read_request_report report;
    int rc = ft_tt_udrv_config_max_read_request_set(udrv->pci.sysfs_path,
                                                    FT_TT_UDRV_MRRS_BYTES_DEFAULT, &report);

    if (rc != FT_TT_UDRV_OK) {
        /* `bytes_after` is only meaningful once DEVCTL has been read; on the
         * capability-walk failure it is still zero and printing it would state a
         * value nobody observed. */
        if (report.capability_offset != 0u)
            ft_tt_log_error("tt runtime udrv: MRRS NOT SET — %s (errno %d). The card keeps %u "
                            "bytes (DEVCTL code %u). A config-space write is privileged; on a "
                            "box where tt-kmd never ran nothing else sets this",
                            report.failure_stage ? report.failure_stage : "unknown",
                            report.failure_errno, report.bytes_after, report.code_after);
        else
            ft_tt_log_error("tt runtime udrv: MRRS NOT SET and its CURRENT VALUE IS UNKNOWN — "
                            "%s (errno %d). Non-root cannot read or write it; a udev rule with "
                            "`setpci CAP_EXP+8.w=5000:7000` is the root-side answer",
                            report.failure_stage ? report.failure_stage : "unknown",
                            report.failure_errno);
        return;
    }
    ft_tt_log_error("tt runtime udrv: MRRS %u -> %u bytes (%s)", report.bytes_before,
                    report.bytes_after,
                    report.written ? "WRITTEN by us" : "already correct, nothing written");
}

/**
 * Memory decode — `pci_enable_device()`'s job, and nothing on a cold host does it.
 *
 * MEASURED ON a second card host, 2026-08-18, on a board that has never had tt-kmd:
 * PCI_COMMAND reads 0x0100, the BAR0 mmap SUCCEEDS, and all 1024 words of it
 * read 0xFFFFFFFF. The ARC bring-up then failed at its first NoC read. On the
 * p150a this never showed, because `pci_disable_device()` leaves the bit set and
 * every reading there was taken after tt-kmd had run at least once.
 *
 * BEST EFFORT, exactly as MRRS is and for the same reason — the write is
 * privileged and the udev rule is the non-root answer. It is loud rather than
 * fatal because the failure it would cause is caught one step later with a
 * better message: BAR0 answering all-ones is what `ft_tt_udrv_arc_locate_queue()`
 * rejects by name.
 */
static void enable_memory_decode(ft_tt_runtime_udrv *udrv)
{
    ft_tt_udrv_memory_decode_report report;
    int rc = ft_tt_udrv_config_memory_decode_enable(udrv->pci.sysfs_path, &report);

    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt runtime udrv: MEMORY DECODE NOT ENABLED — %s (errno %d). "
                        "PCI_COMMAND reads 0x%04x. With bit 1 clear the BAR still maps and "
                        "every word of it reads 0xFFFFFFFF; a config-space write is "
                        "privileged, so this needs root or the udev rule's "
                        "`setpci COMMAND=0002:0002`",
                        report.failure_stage ? report.failure_stage : "unknown",
                        report.failure_errno, report.command_after);
        return;
    }
    ft_tt_log_error("tt runtime udrv: memory decode PCI_COMMAND 0x%04x -> 0x%04x (%s)",
                    report.command_before, report.command_after,
                    report.written ? "WRITTEN by us" : "already enabled, nothing written");
}

/** The ARC read32 the shared telemetry walk runs on. `ft_tt_udrv_arc_noc_read32`
 *  is tt-kmd's own `noc_read32`: any tile-local address on the ARC tile, which
 *  is what the reset-scratch pointers and the CSM both need. */
static int arc_read32(void *user, uint32_t address, uint32_t *out_value)
{
    ft_tt_runtime_udrv *udrv = (ft_tt_runtime_udrv *)user;

    return udrv_status(ft_tt_udrv_arc_noc_read32(&udrv->arc, address, out_value));
}

int ft_tt_runtime_udrv_read_telemetry(ft_tt_runtime_udrv *udrv, const uint16_t *tags,
                                      unsigned count, uint32_t *out_values, bool *out_found)
{
    if (!udrv || !udrv->arch)
        return FT_TT_EINVAL;
    return ft_tt_arc_telemetry_read_tags_via(udrv->arch, arc_read32, udrv, tags, count,
                                             out_values, out_found);
}

/** ASIC_STATE0, SET_WDT_TIMEOUT and POWER_SETTING — `blackhole_init_hardware()`
 *  plus the open-time power request, which is what tt_cdev_open() was doing for
 *  us and what nothing else on the machine will do. */
static int bring_the_card_up(ft_tt_runtime_udrv *udrv)
{
    ft_tt_udrv_arc_probe_report report;
    ft_tt_noc_coord arc_coord;
    int rc = ft_tt_coord_arc(udrv->arch, FT_TT_COORD_TRANSLATED, &arc_coord);

    if (rc != FT_TT_OK)
        return rc;

    /* THE ARC SHARES THE CONTROL BANK'S FIRST WINDOW rather than claiming a
     * third. It aims that path through the same function the bank's own
     * transfers do, so the bank's lookup — which reads the paths' OWN aim state
     * and keeps no second copy of it — sees the ARC's aim immediately and never
     * routes a transfer at a window the ARC has moved. */
    rc = ft_tt_udrv_arc_transport_init(&udrv->arc, &udrv->control.path[0], (uint16_t)arc_coord.x,
                                       (uint16_t)arc_coord.y, &udrv->arc_transport);
    if (rc != FT_TT_UDRV_OK)
        return udrv_status(rc);

    memset(&report, 0, sizeof(report));
    rc = ft_tt_udrv_arc_probe_hardware(&udrv->arc_transport, FT_TT_UDRV_ARC_WATCHDOG_DEFAULT_MS,
                                       ft_tt_ai_clock_wants_maximum(), &report);
    if (rc != FT_TT_UDRV_OK) {
        /* The STAGE leads, because the numbers after it are only meaningful once
         * something was actually read. This message used to open with
         * "boot status 0x00000000 after 0 polls" on a card whose BARs were not
         * even decoding, which reads as a sleeping ARC and sends the reader after
         * the firmware instead of after PCI_COMMAND. */
        ft_tt_log_error("tt runtime udrv: ARC bring-up failed — %s (locate boot status 0x%08x "
                        "after %u polls; ASIC_STATE0 rc=%d, SET_WDT_TIMEOUT rc=%d, "
                        "POWER_SETTING rc=%d)",
                        report.locate.failure_stage ? report.locate.failure_stage
                                                    : "no stage reported",
                        report.locate.boot_status, report.locate.boot_status_polls,
                        report.asic_state0.rc, report.set_wdt_timeout.rc, report.power_setting.rc);
        return udrv_status(rc);
    }
    return FT_TT_OK;
}

/* ================================================================
 * Open and close
 * ================================================================ */

/** The body of the open, so every failure lands on one cleanup path. */
static int runtime_udrv_open_locked(int device_index, ft_tt_runtime_udrv *udrv,
                                    ft_tt_grid_info *out_grid)
{
    ft_tt_grid_info grid;
    int rc = find_card(device_index, udrv);

    if (rc != FT_TT_OK)
        return rc;

    /* Both before the BAR is mapped, which is the order tt-kmd's probe uses.
     * Memory decode goes FIRST: MRRS is probe parity, but without decode the
     * mapping still succeeds and every register reads 0xFFFFFFFF. */
    enable_memory_decode(udrv);
    set_max_read_request_size(udrv);

    rc = udrv_status(ft_tt_udrv_noc_device_open(&udrv->pci, true, &udrv->device));
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt runtime udrv: cannot map BAR0 of %s — the resource files are "
                        "root-owned 0600, so this needs root or a udev rule",
                        udrv->pci.address);
        return rc;
    }
    udrv->device_open = true;

    rc = udrv_status(ft_tt_udrv_claim_namespace_init(&udrv->windows, NULL, udrv->pci.address,
                                                     "tlb2m", FT_TT_UDRV_TLB_2M_WINDOW_COUNT));
    if (rc != FT_TT_OK)
        return rc;

    rc = open_window_bank(udrv, false, "register", ft_tt_udrv_noc_bank_windows_selected(),
                          &udrv->control);
    if (rc != FT_TT_OK)
        return rc;
    udrv->control_open = true;

    /* The bring-up runs on the UC path BEFORE the grid is read, because the
     * telemetry table is what the bring-up's own effect is measured on and
     * because nothing else on this machine sends POWER_SETTING. */
    rc = bring_the_card_up(udrv);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_arc_telemetry_read_grid_via(udrv->arch, arc_read32, udrv, &grid);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt runtime udrv: telemetry read failed: %s", ft_tt_error_string(rc));
        return rc;
    }
    /* A topology we cannot map is a topology we must not probe: a wrong NoC
     * endpoint hangs the chip (spec §7.3). Refuse before any bulk window. */
    rc = ft_tt_coord_check_grid(&grid);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt runtime udrv: grid is not a topology we can map; refusing the card");
        return rc;
    }

    rc = open_window_bank(udrv, true, "bulk", ft_tt_udrv_noc_bank_windows_selected(),
                          &udrv->bulk);
    if (rc != FT_TT_OK)
        return rc;
    udrv->bulk_open = true;
    /* The reader banks are OPTIONAL: a process that cannot claim one runs its reads on the
     * bulk bank, as before. `FT_TT_UDRV_READER_WINDOWS=0` skips them; `FT_TT_UDRV_READER_BANKS`
     * (default 2) is how many threads may each own one. */
    {
        const char *env = getenv("FT_TT_UDRV_READER_WINDOWS");
        const char *banks_env = getenv("FT_TT_UDRV_READER_BANKS");
        const long want = env && *env ? strtol(env, NULL, 10) : 2L;
        long banks = banks_env && *banks_env ? strtol(banks_env, NULL, 10) : 2L;

        if (banks > (long)FT_TT_UDRV_READER_BANKS_MAX)
            banks = (long)FT_TT_UDRV_READER_BANKS_MAX;
        while (want > 0 && want <= (long)FT_TT_UDRV_NOC_BANK_MAX_WINDOWS &&
               (long)udrv->reader_banks < banks &&
               open_window_bank(udrv, true, "reader", (uint32_t)want,
                                &udrv->reader[udrv->reader_banks]) == FT_TT_OK)
            udrv->reader_banks++;
    }
    /* THE COUNTS ARE ECHOED, NOT ASSUMED. `FT_TT_UDRV_WINDOWS` is the knob the
     * aim ladder is scored on, and a knob that reads as armed and is not is
     * this project's most expensive instrument defect — so what was ASKED FOR
     * and what was ACHIEVED are both on the line, and the first window index of
     * each bank names itself the way the single-window line used to. */
    ft_tt_log_error("tt runtime udrv: %s, NO VENDOR DRIVER — %u register windows from %u (UC) "
                    "and %u bulk windows from %u (WC), %u asked for, index %u reserved for the "
                    "vendor driver, %u Tensix in %u columns",
                    udrv->pci.address, udrv->control.window_count,
                    udrv->control.path[0].window_id, udrv->bulk.window_count,
                    udrv->bulk.path[0].window_id, ft_tt_udrv_noc_bank_windows_selected(),
                    udrv->arch->tlb.kernel_window_index, grid.tensix_core_count,
                    grid.tensix_column_count);
    *out_grid = grid;
    return FT_TT_OK;
}

int ft_tt_runtime_udrv_open(int device_index, ft_tt_runtime_udrv **out_udrv,
                            ft_tt_grid_info *out_grid, ft_tt_noc_access *out_noc,
                            ft_tt_noc_access *out_bulk_dram, ft_tt_noc_access *out_bulk_l1)
{
    ft_tt_runtime_udrv *udrv = NULL;
    int rc;

    if (!out_udrv || !out_grid || !out_noc || !out_bulk_dram || !out_bulk_l1)
        return FT_TT_EINVAL;

    udrv = calloc(1, sizeof(*udrv));
    if (!udrv)
        return FT_TT_ENOMEM;

    rc = runtime_udrv_open_locked(device_index, udrv, out_grid);
    if (rc != FT_TT_OK) {
        ft_tt_runtime_udrv_close(udrv);
        return rc;
    }

    /* Every `user` points into the HEAP handle, not into any caller's struct,
     * so the accessors survive the caller copying its runtime by value. That is
     * the one thing the tt-kmd path has to re-bind by hand after the copy. */
    out_noc->write = ft_tt_udrv_noc_bank_access_write;
    out_noc->read = ft_tt_udrv_noc_bank_access_read;
    out_noc->user = &udrv->control;
    out_bulk_dram->write = bulk_dram_write;
    out_bulk_dram->read = bulk_dram_read;
    out_bulk_dram->user = &udrv->bulk;
    out_bulk_l1->write = bulk_l1_write;
    out_bulk_l1->read = bulk_l1_read;
    out_bulk_l1->user = &udrv->bulk;

    *out_udrv = udrv;
    return FT_TT_OK;
}

void ft_tt_runtime_udrv_close(ft_tt_runtime_udrv *udrv)
{
    if (!udrv)
        return;
    /* The histogram goes out BEFORE the banks are torn down, because closing a
     * bank zeroes it. It is a no-op unless FT_TT_UDRV_AIM_HISTOGRAM was set. */
    ft_tt_udrv_noc_bank_histogram_log(&udrv->control);
    ft_tt_udrv_noc_bank_histogram_log(&udrv->bulk);
    /* Same rule for the reuse ladder, and a no-op unless FT_TT_UDRV_AIM_REUSE
     * was set. It is what sizes the bank, so it prints its own self-check. */
    ft_tt_udrv_noc_bank_reuse_log(&udrv->control);
    ft_tt_udrv_noc_bank_reuse_log(&udrv->bulk);
    while (udrv->reader_banks > 0u)
        (void)ft_tt_udrv_noc_bank_close(&udrv->reader[--udrv->reader_banks]);
    if (udrv->bulk_open)
        (void)ft_tt_udrv_noc_bank_close(&udrv->bulk);
    if (udrv->control_open)
        (void)ft_tt_udrv_noc_bank_close(&udrv->control);
    if (udrv->device_open)
        (void)ft_tt_udrv_noc_device_close(&udrv->device);
    free(udrv);
}

int ft_tt_runtime_udrv_reader_access(ft_tt_runtime_udrv *udrv, uint32_t index,
                                     ft_tt_noc_access *out_dram)
{
    if (!udrv || !out_dram)
        return FT_TT_EINVAL;
    if (index >= udrv->reader_banks)
        return FT_TT_EUNAVAIL;
    out_dram->write = bulk_dram_write;
    out_dram->read = bulk_dram_read;
    out_dram->user = &udrv->reader[index];
    return FT_TT_OK;
}

int ft_tt_runtime_udrv_reader_l1_access(ft_tt_runtime_udrv *udrv, uint32_t index,
                                        ft_tt_noc_access *out_l1)
{
    if (!udrv || !out_l1)
        return FT_TT_EINVAL;
    if (index >= udrv->reader_banks)
        return FT_TT_EUNAVAIL;
    out_l1->write = bulk_l1_write;
    out_l1->read = bulk_l1_read;
    out_l1->user = &udrv->reader[index];
    return FT_TT_OK;
}

const char *ft_tt_runtime_udrv_pci_address(const ft_tt_runtime_udrv *udrv)
{
    return udrv ? udrv->pci.address : "";
}

int ft_tt_runtime_udrv_window_stats_read(const ft_tt_runtime_udrv *udrv,
                                    ft_tt_runtime_window_stats *out)
{
    if (!out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));
    if (!udrv)
        return FT_TT_EINVAL;

    out->register_read = ft_tt_udrv_noc_bank_bytes_read(&udrv->control);
    out->register_written = ft_tt_udrv_noc_bank_bytes_written(&udrv->control);
    out->register_aims = ft_tt_udrv_noc_bank_aim_count(&udrv->control);
    out->bulk_read = ft_tt_udrv_noc_bank_bytes_read(&udrv->bulk);
    out->bulk_written = ft_tt_udrv_noc_bank_bytes_written(&udrv->bulk);
    out->bulk_aims = ft_tt_udrv_noc_bank_aim_count(&udrv->bulk);
    out->aim_failures = ft_tt_udrv_noc_bank_aim_failures(&udrv->control) +
                        ft_tt_udrv_noc_bank_aim_failures(&udrv->bulk);
    out->aim_seconds = ft_tt_udrv_noc_bank_aim_seconds(&udrv->control) +
                       ft_tt_udrv_noc_bank_aim_seconds(&udrv->bulk);
    out->aim_verify_seconds = ft_tt_udrv_noc_bank_aim_verify_seconds(&udrv->control) +
                              ft_tt_udrv_noc_bank_aim_verify_seconds(&udrv->bulk);
    out->register_windows = udrv->control.window_count;
    out->bulk_windows = udrv->bulk.window_count;
    out->register_hits = udrv->control.hits;
    out->register_misses = udrv->control.misses;
    out->register_evictions = udrv->control.evictions;
    out->bulk_hits = udrv->bulk.hits;
    out->bulk_misses = udrv->bulk.misses;
    out->bulk_evictions = udrv->bulk.evictions;
    out->misaims_injected = udrv->control.misaims_injected + udrv->bulk.misaims_injected;
    out->audit_failures = udrv->control.audit_failures + udrv->bulk.audit_failures;
    out->register_endpoints = ft_tt_udrv_noc_bank_endpoint_count(&udrv->control);
    out->bulk_endpoints = ft_tt_udrv_noc_bank_endpoint_count(&udrv->bulk);
    return FT_TT_OK;
}
