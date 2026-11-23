/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_device_reset.h
 * @brief The device- and bus-level rungs of the recovery ladder: tt-kmd's `RESET_DEVICE`
 *   ioctl (rungs 3 and 4) and a PCIe Secondary Bus Reset of the bridge the card is alone
 *   behind (rung 5).
 *
 * These are whole-card operations. Nothing here targets a core; they are here because the
 * failure they are aimed at — a Tensix that will not boot firmware again — has so far only
 * ever been cleared by a host reboot, and a reboot's active ingredient is PERST#, not the
 * kernel restarting.
 *
 * ---------------------------------------------------------------------------
 * WHY RUNG 5 IS SAFE ON THIS HOST, AND HOW THAT IS ENFORCED
 * ---------------------------------------------------------------------------
 * A Secondary Bus Reset resets EVERY device behind the bridge it is issued on. On this
 * machine the card was alone behind root port `00:1c.6` while all eight K80s sat under
 * `1c.0`, `1c.3`, `1c.4` and `1c.5` — so an SBR on `1c.6` touches the card and nothing
 * else. That is a property of this machine's topology, not of the code, so the code
 * REFUSES rather than assumes: `ft_tt_pcie_topology_resolve()` derives the bridge FROM the
 * card (it is never named by the caller), requires that the bridge has exactly one child,
 * and requires that child to be the Tenstorrent card. Anything else is refused with the
 * reason. Nobody should be able to reset a bridge with K80s under it by typo.
 *
 * It is also only viable because the firmware flash set `pcie_bar4_size = 0`. With the
 * stock 32 GiB BAR4 request, re-enumeration on a 36-bit host cannot place the BAR and the
 * device does not come back (`docs/p150a-flashing.md`).
 *
 * ---------------------------------------------------------------------------
 * ON THE ONE PREVIOUS ATTEMPT
 * ---------------------------------------------------------------------------
 * Flags 4 and 5 were tried on 2026-08-11 against a HUNG NoC and neither recovered it. That
 * is not precedent for this failure: the ARC was dead then, and both paths need a working
 * ARC (flag 5 literally sends it a message). Here the ARC answers telemetry.
 *
 * @status Runs on the p150a test card. Linux-only by construction — every path is
 *   sysfs or a tt-kmd ioctl. MEASURED 2026-08-12: `ASIC_RESET` + `ft_tt_pcie_reenumerate()`
 *   recovered all three wedged Tensixes without rebooting the host, and the SBR ran clean
 *   but left the Tensix grid byte-identical, so it resets the PCIe layer and not the compute
 *   grid. `ASIC_DMC_RESET` is written and NOT exercised — rung
 *   3 already recovered the cores, and it sits above rung 3 in the ladder.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_DEVICE_RESET_H
#define FT_TT_DEVICE_RESET_H

#include <stdbool.h>
#include <stdint.h>

/** Longest "dddd:bb:dd.f" plus terminator. */
#define FT_TT_PCI_ADDRESS_MAX 16u

/** `tenstorrent_reset_device_in.flags` (the kernel driver's reset-device ioctl). The ABI header
 *  `tt/ft_tt_kmd_abi.h` deliberately carries no struct for this ioctl so that nothing can
 *  issue it by accident; this module is the one place that may, so the flags and the
 *  payload live here. */
typedef enum ft_tt_device_reset_flag {
    FT_TT_RESET_DEVICE_RESTORE_STATE = 0,
    FT_TT_RESET_DEVICE_RESET_PCIE_LINK = 1,
    FT_TT_RESET_DEVICE_CONFIG_WRITE = 2,
    FT_TT_RESET_DEVICE_USER_RESET = 3,
    FT_TT_RESET_DEVICE_ASIC_RESET = 4,
    FT_TT_RESET_DEVICE_ASIC_DMC_RESET = 5,
    FT_TT_RESET_DEVICE_POST_RESET = 6
} ft_tt_device_reset_flag;

/** What one `RESET_DEVICE` sequence did. `driver_result` is the ioctl's own `out.result`,
 *  which is 0 for success — the driver reports failure IN the payload, not as -errno. */
typedef struct ft_tt_device_reset_report {
    ft_tt_device_reset_flag flag;
    int ioctl_errno;               /**< 0, or errno from the reset ioctl */
    uint32_t driver_result;
    bool post_reset_issued;
    unsigned post_reset_attempts;  /**< the marker is not clear the instant the reset lands */
    int post_reset_errno;
    uint32_t post_reset_driver_result;
    bool chardev_present_after;    /**< /dev/tenstorrent/0 still there afterwards */
} ft_tt_device_reset_report;

/** What re-enumerating the card through sysfs did. Used both to COMPLETE a RESET_DEVICE
 *  (whose in-place completion does not finish the job on this host — see the header of
 *  ft_tt_pcie_reenumerate) and as the second half of an SBR. */
typedef struct ft_tt_pcie_reenumeration_report {
    bool removed;
    bool rescanned;
    bool device_returned;
    bool memory_decode_enabled;
    bool driver_bound;
    bool chardev_present;
    uint64_t bar0_length_after;
    const char *failure_stage;
    int failure_errno;
} ft_tt_pcie_reenumeration_report;

/** The bridge a card sits behind, and whether an SBR on it is defensible. */
typedef struct ft_tt_pcie_topology {
    char card[FT_TT_PCI_ADDRESS_MAX];
    char bridge[FT_TT_PCI_ADDRESS_MAX];
    unsigned child_count;
    char unexpected_child[FT_TT_PCI_ADDRESS_MAX]; /**< first child that is not the card */
    uint16_t card_vendor_id;
    uint16_t card_device_id;
    bool safe;               /**< exactly one child and it is the Tenstorrent card */
    const char *refusal;     /**< why not; NULL when safe */
} ft_tt_pcie_topology;

/** What the SBR did, step by step, so a failure names the step it stopped at. */
typedef struct ft_tt_pcie_sbr_report {
    ft_tt_pcie_topology topology;
    bool config_saved;
    uint64_t bar0_length_before;
    uint64_t bar0_length_after;
    bool device_removed;
    uint16_t bridge_control_before;
    uint16_t bridge_control_asserted;
    uint16_t bridge_control_after;
    bool rescanned;
    bool device_returned;
    bool driver_bound;
    bool chardev_present;
    bool config_matches_saved;   /**< the re-enumerated device's identity words match */
    const char *failure_stage;   /**< NULL on success */
    int failure_errno;
} ft_tt_pcie_sbr_report;

/**
 * Issue one `RESET_DEVICE` flag, then the `POST_RESET` half that re-initialises the
 * hardware.
 *
 * The caller must have CLOSED every handle to the device first: the ioctl zaps VMAs and
 * reclaims TLB windows, so a live session's mapped windows are invalid the moment it
 * returns.
 *
 * @param chardev_path  e.g. "/dev/tenstorrent/0"
 * @param flag          FT_TT_RESET_DEVICE_ASIC_RESET or _ASIC_DMC_RESET
 * @param out           zeroed and filled
 * @return              FT_TT_OK once the sequence ran; FT_TT_EINVAL on a bad argument;
 *                      FT_TT_EIO if the device could not be opened
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_device_reset_issue(const char *chardev_path, ft_tt_device_reset_flag flag,
                             ft_tt_device_reset_report *out);

/** Human-readable account of a RESET_DEVICE sequence. */
void ft_tt_device_reset_print(const ft_tt_device_reset_report *report);

/**
 * Remove the card from the PCI bus and rescan its bridge, so the kernel re-enumerates it
 * and the driver re-probes from scratch.
 *
 * **This is the completion half of a RESET_DEVICE on this host, and it is not optional.**
 * MEASURED 2026-08-12: after `ASIC_RESET`, the driver's own `POST_RESET` first returned
 * FAILED (it was issued before the card had come out of reset, so the reset marker still
 * read as set); once it succeeded, the card was still left with memory decode DISABLED and
 * an ARC that answered `0xffffffff` to every telemetry read, and tt-kmd logged `Failed to
 * set initial power state: -5` on every open. A remove + bridge rescan restored all of it:
 * memory decode on, driver bound, ARC answering, all 140 cores reading the power-on reset
 * default. Leaving this out is how someone concludes the card is dead when it is one sysfs
 * write from working.
 *
 * Needs root. Rescanning a bridge is not a reset of anything: it enumerates.
 */
int ft_tt_pcie_reenumerate(const char *card_bdf, ft_tt_pcie_reenumeration_report *out);

/** Can this process complete a re-enumeration? `FT_TT_OK`, or `FT_TT_EPERM` if the sysfs
 *  control nodes are not writable.
 *
 *  CALL THIS BEFORE ANY CARD-LEVEL RESET. Every one of them leaves the card
 *  reset-but-not-re-enumerated, and only the remove + rescan finishes the job. Finding out
 *  AFTERWARDS that this process cannot do it leaves the card dead until a human runs the
 *  rescan by hand — measured on 2026-08-12, errno 13 at "removing the card from the PCI
 *  bus", card unusable until re-enumerated manually. Uses `access()` on the real nodes
 *  rather than `geteuid()`, so a capability or ACL that grants it without uid 0 is
 *  honoured, and so the answer is about the files that will actually be written. */
int ft_tt_pcie_reenumerate_is_permitted(const char *card_bdf);

/** What re-enumeration recovered, line by line. */
void ft_tt_pcie_reenumeration_print(const ft_tt_pcie_reenumeration_report *report);

/**
 * Derive the bridge `card_bdf` sits behind and decide whether an SBR on it is defensible.
 *
 * Reads only sysfs. The bridge is DERIVED from the card's device path — there is no way to
 * name a bridge from outside this module, which is the point.
 *
 * @param card_bdf  "0000:12:00.0"
 * @param out       zeroed and filled; `safe` and `refusal` carry the verdict
 * @return          FT_TT_OK once the topology was read; FT_TT_EIO if sysfs would not
 *                  answer. A REFUSAL is a successful read with `safe == false`
 */
int ft_tt_pcie_topology_resolve(const char *card_bdf, ft_tt_pcie_topology *out);

/** One line per fact: bridge, children, and the verdict. */
void ft_tt_pcie_topology_print(const ft_tt_pcie_topology *topology);

/**
 * Rung 5 — remove the card, pulse Secondary Bus Reset on its bridge, rescan, and verify
 * the device and its BARs came back.
 *
 * Refuses unless `ft_tt_pcie_topology_resolve()` says the bridge has exactly one child and
 * that child is the Tenstorrent card. Needs root: it writes PCI config space through
 * sysfs.
 *
 * Config space for both the card and the bridge is saved to `save_directory` before
 * anything is written — a successful reset erases the evidence otherwise. The card's
 * config is NOT written back: the kernel re-enumerates and re-assigns BARs on rescan, and
 * a saved copy pushed over that would fight it. The saved copy is compared instead. The
 * BRIDGE's Bridge Control register, the one register this function modifies, is restored
 * exactly.
 *
 * @param card_bdf        "0000:12:00.0"
 * @param save_directory  where the two config-space dumps go; NULL skips saving, which is
 *                        allowed and recorded but not recommended
 * @param out             zeroed and filled
 * @return                FT_TT_OK once the sequence ran to a verdict; FT_TT_EPERM if the
 *                        topology check refused; FT_TT_EIO on a sysfs failure
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_pcie_secondary_bus_reset(const char *card_bdf, const char *save_directory,
                                   ft_tt_pcie_sbr_report *out);

/** Step-by-step account of an SBR attempt. */
void ft_tt_pcie_sbr_print(const ft_tt_pcie_sbr_report *report);

#endif /* FT_TT_DEVICE_RESET_H */
