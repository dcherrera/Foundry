/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_reset.h
 * @brief The RESET half of the Foundry userspace driver: an in-place ASIC reset of a
 *   Blackhole card, its PCI config-space snapshot/restore, and the PCIe link retrain that
 *   must follow — all of it through `/sys/bus/pci/devices/<bdf>/config`, with no vendor
 *   kernel module and no ioctl.
 *
 *   This is the one loss in `ft_tt_udrv.h` §5 that was reasoned to the register and never
 *   executed. It is the highest risk in Phase 12 because the card is moving to a board that
 *   will not have tt-kmd: the driver can talk to the card long before it can reliably get it
 *   back, and the gap between those two dates is the exposure.
 *
 * ---------------------------------------------------------------------------
 * WHAT AN ASIC RESET IS ON THIS PART
 * ---------------------------------------------------------------------------
 * Three PCI CONFIG writes and nothing else (tt-kmd `blackhole.c:blackhole_reset()`,
 * `pcie.c:set_reset_marker()`, `pcie.c:pcie_timer_interrupt()`):
 *
 *   1. `PCI_COMMAND |= PCI_COMMAND_PARITY`  (config 0x04, bit 6) — the RESET MARKER. The
 *      card clears it when it comes out of reset, and that is the only completion signal
 *      there is.
 *   2. config dword `0x934 <- 0x00000001`   — INTERFACE_TIMER_TARGET
 *   3. config dword `0x930 <- 0x00000011`   — INTERFACE_TIMER_EN | INTERFACE_FORCE_PENDING
 *
 * Config space is 4096 bytes on this board, so all three are `pwrite()` offsets. There is no
 * ARC message in the path, which is what makes this reachable before the ARC transport
 * exists. ORDER IS LOAD-BEARING: 0x934 arms the timer target, 0x930 fires it. Firing before
 * arming resets against whatever target the previous owner left.
 *
 * ---------------------------------------------------------------------------
 * WHY IN-PLACE SNAPSHOT/RESTORE AND NOT `remove` + `rescan`
 * ---------------------------------------------------------------------------
 * `pci_save_state`/`pci_restore_state` have no sysfs equivalent, so a userspace in-place
 * reset must take its own config snapshot and write it back. The alternative is
 * `remove` + `rescan`, which is what `ft_tt_core recover --rung 3` does today. Both were
 * considered; in place wins, for reasons that are about the machine the card is MOVING to:
 *
 *   - **A rescan re-runs BAR assignment, and that can fail permanently.** This card only
 *     re-enumerates at all because its firmware flash set `pcie_bar4_size = 0`; with the
 *     stock 32 GiB BAR4 request a 36-bit host cannot place the BAR and the device does not
 *     come back (`docs/p150a-flashing.md`, quoted in `tools/ft_tt_device_reset.h`). An
 *     in-place restore never re-assigns anything — it writes back the addresses the PCI core
 *     already chose — so it cannot fail that way.
 *   - **`remove` is not scopeable.** It tears the device out from under every consumer and
 *     needs `/sys/bus/pci/rescan`, a BUS-WIDE control. In-place touches one device's config.
 *   - **It is what the vendor driver does.** tt-kmd's ASIC_RESET is in-place; the remove +
 *     rescan in rung 3 is Foundry's own addition, and this file's @status records why it was
 *     needed (a tt-kmd bug, not a hardware requirement).
 *
 * `remove` + `rescan` is kept as an explicit fallback in the tool, not deleted, because if
 * the ASIC reset ever fails to bring the card back it is the only other lever.
 *
 * ---------------------------------------------------------------------------
 * WHAT `pci_restore_state` ACTUALLY RESTORES — replicated, not approximated
 * ---------------------------------------------------------------------------
 * The kernel's order is not cosmetic and this module matches it exactly:
 *
 *   1. the PCI Express capability's seven writable words, in save order:
 *      DEVCTL (+0x08, which is where MRRS lives), LNKCTL (+0x10), SLTCTL (+0x18),
 *      RTCTL (+0x1C), DEVCTL2 (+0x28), LNKCTL2 (+0x30), SLTCTL2 (+0x38)
 *   2. header dwords 10..15 (config 0x28..0x3F)
 *   3. header dwords 4..9 (config 0x10..0x27) — **the BARs, before the command register**
 *   4. header dwords 0..3 (config 0x00..0x0F) — dword 1 is COMMAND, which re-enables memory
 *      decode, and it must land after the BARs it is enabling
 *
 * Restoring the command register before the BARs would enable a decoder aimed at address
 * zero. `ft_tt_udrv_config_restore_report.order` records what was written and in what
 * sequence so that this is a checkable property and not a comment.
 *
 * ---------------------------------------------------------------------------
 * THE LINK DROPS TO GEN1 EVERY TIME
 * ---------------------------------------------------------------------------
 * Measured repeatedly: after any card-level reset the link comes back at
 * 2.5 GT/s and every host-involved timing silently halves. The fix is one config write to
 * the UPSTREAM BRIDGE — `LNKCTL |= Retrain Link` (bit 5), i.e. what
 * `setpci -s $BRIDGE CAP_EXP+10.w=0020:0020` does — then poll the bridge's LNKSTA until the
 * Link Training bit clears. It is a separate function from the reset because it is a
 * different device's config space, and because the link must be checked at BOTH ends of a
 * round whether or not anything was reset.
 *
 * @status ON SILICON 2026-08-17, p150a at 0000:12:00.0, with tt-kmd loaded as the safety
 *   net. The in-place path recovered a DELIBERATELY WEDGED core with no PCI remove and no
 *   rescan: core 101 at translated (4,9), wedged by the 5-destination multicast relaunch arm,
 *   fingerprint `soft reset 0x00047000: BR=running NC/T0/T1/T2=held`. **`probe --write`
 *   FAILED on it before ("firmware boot rc=-4, 0/4 runs") and PASSED 4/4 byte-exact after**,
 *   while its five neighbours passed throughout. `probe all` 140/140 afterwards.
 *
 *   Numbers from that run, none of which were predictable from source:
 *     - the card cleared its own reset marker in **115 ms, 3 polls**
 *     - **the ASIC reset clears only 6 of the 23 saved registers**: BAR0 (both dwords),
 *       BAR2's upper dword, `PCI_COMMAND` (to 0x0000 — memory decode OFF), the cacheline
 *       dword, and `DEVCTL` (MRRS back to 4096 from the restore's 0x5110). The other 17 read
 *       back unchanged and were recorded `skipped`. Header dwords 10-15 survive intact.
 *     - the link dropped to **2.5 GT/s** exactly as documented, and the bridge retrain took
 *       it back to 5 GT/s x1 in **11 ms**
 *
 *   That settles `ft_tt_udrv.h` §5(e)'s discriminating experiment: the wedged state is ASIC
 *   configuration cleared at reset, NOT something that survives until re-enumeration. A
 *   narrower per-core equivalent may therefore exist — but the only register that reaches the
 *   Tensix backend is still permanently banned, so this is a direction, not a plan.
 *
 *   And it explains why rung 3 needed the remove + rescan, which was read as a hardware
 *   requirement and is not one: tt-kmd's `POST_RESET` clears `needs_hw_init` BEFORE deciding
 *   whether it can restore, so a POST_RESET issued while the reset
 *   marker is still set loses the restore permanently and the next POST_RESET is a no-op.
 *   This module waits for the marker to clear before restoring. That is the whole difference.
 * @issues
 *   - The ARC probe sequence (ASIC_STATE0, SET_WDT_TIMEOUT) is NOT part of this module and
 *     has no transport yet — `ft_tt_udrv_arc_msg.h`. MRRS is covered, because it lives in
 *     DEVCTL and the config restore rewrites it. On the card host the ARC came back live with no
 *     handshake from us at all — the probe's chardev open sends tt-kmd's POWER_SETTING. **On a
 *     board with no tt-kmd nothing sends either message, and this round did not test that.**
 *     The reset needs neither; whether the card is fully initialised afterwards is untested.
 *   - `tenstorrent_vma_zap` has no userspace equivalent. The discipline that replaces it is
 *     the caller's: hold the card lock, munmap every aperture, reset, re-map. A second
 *     process that mapped the BAR behind our back keeps a live mapping across the reset.
 * @todo None
 */

#ifndef FT_TT_UDRV_RESET_H
#define FT_TT_UDRV_RESET_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** Config space is 4096 bytes on this board; the standard header is the first 64. */
#define FT_TT_UDRV_CONFIG_HEADER_DWORDS 16u

/** DEVCTL, LNKCTL, SLTCTL, RTCTL, DEVCTL2, LNKCTL2, SLTCTL2 — `pci_save_pcie_state()`. */
#define FT_TT_UDRV_PCIE_SAVED_WORDS 7u

/** Every dword and word one restore may write, so the order is recordable in full:
 *  7 capability words + 16 header dwords. */
#define FT_TT_UDRV_RESTORE_STEPS_MAX 23u

/** PCI capability id of the PCI Express capability. */
#define FT_TT_UDRV_CAP_ID_EXPRESS 0x10u

/** `PCI_COMMAND` bit 6. tt-kmd uses it as the reset marker; so do we. */
#define FT_TT_UDRV_COMMAND_PARITY 0x0040u

/** `LNKCTL` bit 5 — Retrain Link. Write-1, self-clearing. */
#define FT_TT_UDRV_LNKCTL_RETRAIN 0x0020u

/** `LNKSTA` bit 11 — Link Training. Set while the link is retraining. */
#define FT_TT_UDRV_LNKSTA_TRAINING 0x0800u

/** Longest "<device_dir>/config" this module will build. */
#define FT_TT_UDRV_RESET_PATH_MAX 320u

/** Encoded current link speed, `LNKSTA` bits 3:0. 1 = 2.5 GT/s, 2 = 5, 3 = 8, ... */
#define FT_TT_UDRV_LINK_SPEED_2_5GT 1u
#define FT_TT_UDRV_LINK_SPEED_5GT 2u

/**
 * A snapshot of everything a userspace driver must write back after an in-place reset.
 *
 * `pcie_capability_offset` is 0 when the device has no PCI Express capability, which is a
 * refusal and not a tolerable case: without it MRRS cannot be restored.
 *
 * DEVCTL2, LNKCTL2 and SLTCTL2 exist only at capability VERSION 2 and above. On a version-1
 * capability those offsets belong to whatever follows in config space, so they are recorded
 * as absent and never written — `pcie_word_present` says which of the seven are real. Our
 * board is version 2; a stranger's card need not be, and writing three words past the end of
 * its capability would corrupt the next one.
 */
typedef struct ft_tt_udrv_config_snapshot {
    uint32_t header[FT_TT_UDRV_CONFIG_HEADER_DWORDS]; /**< config 0x00..0x3F */
    uint16_t pcie_word[FT_TT_UDRV_PCIE_SAVED_WORDS];  /**< in save order, see the header */
    bool pcie_word_present[FT_TT_UDRV_PCIE_SAVED_WORDS];
    unsigned pcie_capability_offset;                  /**< 0 = not found */
    unsigned pcie_capability_version;                 /**< PCI_EXP_FLAGS bits 3:0 */
    uint16_t vendor_id;
    uint16_t device_id;
    bool valid;
} ft_tt_udrv_config_snapshot;

/** One entry of a config-space write JOURNAL. Both the restore and the reset keep one,
 *  because in both of them the ORDER of the writes is the load-bearing property and a
 *  journal is the only way a test can see it — the final contents of config space are
 *  identical whatever order they were written in. */
typedef struct ft_tt_udrv_config_step {
    unsigned offset;   /**< config-space byte offset written */
    uint32_t value;    /**< the value written */
    unsigned width;    /**< 2 for a capability word, 4 for a header dword */
    bool skipped;      /**< the register already held `value`, so nothing was written */
} ft_tt_udrv_config_step;

/** What a restore did, in order, and whether the card agreed afterwards. */
typedef struct ft_tt_udrv_config_restore_report {
    ft_tt_udrv_config_step step[FT_TT_UDRV_RESTORE_STEPS_MAX];
    unsigned step_count;
    unsigned written;                /**< steps that actually issued a write */
    unsigned mismatches;             /**< registers that did not read back as intended */
    unsigned first_mismatch_offset;  /**< meaningful only when `mismatches` is nonzero */
    const char *failure_stage;       /**< NULL on success */
    int failure_errno;
} ft_tt_udrv_config_restore_report;

/** The most writes any reset flavour here issues: marker + timer target + timer control. */
#define FT_TT_UDRV_RESET_STEPS_MAX 3u

/** What the three-write ASIC reset did. `step` is the journal, in issue order — the marker
 *  must come first and the timer TARGET must precede the timer CONTROL. */
typedef struct ft_tt_udrv_reset_report {
    ft_tt_udrv_config_step step[FT_TT_UDRV_RESET_STEPS_MAX];
    unsigned step_count;
    uint16_t command_before;
    uint16_t command_with_marker;
    bool marker_set;
    bool timer_target_written;
    bool timer_control_written;
    const char *failure_stage;   /**< NULL on success */
    int failure_errno;
} ft_tt_udrv_reset_report;

/** What waiting for the card to come back saw. */
typedef struct ft_tt_udrv_reset_wait_report {
    unsigned elapsed_ms;
    unsigned polls;
    uint16_t vendor_id_last;
    uint16_t command_last;
    bool vendor_id_returned;   /**< config reads answer with the Tenstorrent vendor id */
    bool marker_cleared;       /**< the card cleared PCI_COMMAND.PARITY itself */
    bool timed_out;
} ft_tt_udrv_reset_wait_report;

/** A PCIe link's negotiated state, read from LNKSTA. */
typedef struct ft_tt_udrv_link_status {
    unsigned speed_code;   /**< LNKSTA bits 3:0 */
    unsigned width;        /**< LNKSTA bits 9:4 */
    bool training;
} ft_tt_udrv_link_status;

/** What a retrain attempt did. */
typedef struct ft_tt_udrv_link_retrain_report {
    ft_tt_udrv_link_status before;
    ft_tt_udrv_link_status after;
    bool retrain_requested;
    bool training_completed;   /**< the Link Training bit cleared within the timeout */
    unsigned elapsed_ms;
    const char *failure_stage; /**< NULL on success */
    int failure_errno;
} ft_tt_udrv_link_retrain_report;

/* ------------------------------------------------------------------------ *
 * Raw config-space access. `device_dir` is a sysfs device directory such as
 * "/sys/bus/pci/devices/0000:12:00.0" — or ANY directory containing a `config`
 * file, which is what makes every function below testable off-hardware.
 * ------------------------------------------------------------------------ */

/**
 * Read one 32-bit config-space dword.
 *
 * @param device_dir  directory holding the `config` file
 * @param offset      byte offset, must be 4-byte aligned
 * @param out_value   the dword, host byte order (config space is little-endian and so is
 *                    every host this runs on; the read is byte-wise and assembled, so the
 *                    result does not depend on the host's alignment rules)
 * @return            FT_TT_UDRV_OK, FT_TT_UDRV_EINVAL on a bad argument, FT_TT_UDRV_EIO if
 *                    the file would not answer (errno preserved)
 *
 * Thread-safe: yes (opens its own fd)   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_config_read_dword(const char *device_dir, unsigned offset, uint32_t *out_value);

/** Read one 16-bit config-space word. Offset must be 2-byte aligned. */
int ft_tt_udrv_config_read_word(const char *device_dir, unsigned offset, uint16_t *out_value);

/** Write one 32-bit config-space dword. Needs write access to `config` (root, or an ACL). */
int ft_tt_udrv_config_write_dword(const char *device_dir, unsigned offset, uint32_t value);

/** Write one 16-bit config-space word. */
int ft_tt_udrv_config_write_word(const char *device_dir, unsigned offset, uint16_t value);

/**
 * Walk the PCI capability list and return the byte offset of one capability.
 *
 * The walk is BOUNDED and refuses a malformed list rather than following it: a capability
 * pointer that is not 4-byte aligned, that points below 0x40, that points outside the
 * 256-byte legacy config window, or that revisits an offset already seen, is a refusal. A
 * card that comes back from a reset half-initialised presents exactly those shapes, and an
 * unbounded walk on one spins forever.
 *
 * @param device_dir      directory holding the `config` file
 * @param capability_id   e.g. FT_TT_UDRV_CAP_ID_EXPRESS
 * @param out_offset      byte offset of the capability header
 * @return                FT_TT_UDRV_OK, FT_TT_UDRV_ENODEV if the list is well-formed and
 *                        simply does not contain it, FT_TT_UDRV_EIO on a malformed list or
 *                        an unreadable file
 */
int ft_tt_udrv_config_find_capability(const char *device_dir, unsigned capability_id,
                                      unsigned *out_offset);

/* ------------------------------------------------------------------------ *
 * MRRS — Max_Read_Request_Size, DEVCTL bits 14:12
 *
 * `pcie_set_readrq(pdev, 4096)` is step 1 of tt-kmd's blackhole_init_hardware(),
 * and the only step in that probe that is not an ARC message. It reads 4096 on
 * any box where the module has ever run — `pci_disable_device()` does not clear
 * DEVCTL, so the value outlives an `rmmod` — and our own config restore rewrites
 * it after a reset. **On a machine where tt-kmd NEVER RAN there is nobody to set
 * it**: it is whatever firmware or the BIOS left, and a bring-up that inherits
 * that inherits it silently. That is the case these functions exist for, and it
 * is the one case this bank can only SIMULATE, by writing a wrong value first.
 *
 * HONEST SCOPE, so nothing is claimed here that this does not do: MRRS bounds the
 * read requests the DEVICE issues upstream. Our data path never has the card read
 * host memory (`ft_tt_udrv.h` §4 — dropping fw_log removed the only such reader),
 * so on the shipping path this is PARITY WITH THE VENDOR PROBE, not a throughput
 * lever. It is set because leaving a documented probe step undone on the one
 * machine that has no probe is how a difference nobody can measure becomes a
 * difference nobody can explain.
 * ------------------------------------------------------------------------ */

/** DEVCTL's Max_Read_Request_Size field. */
#define FT_TT_UDRV_MRRS_SHIFT 12u
#define FT_TT_UDRV_MRRS_MASK 0x7000u

/** Code 5 = 4096 bytes, the largest the field can express and what tt-kmd asks for. */
#define FT_TT_UDRV_MRRS_CODE_MAX 5u

/** The size tt-kmd's probe requests. */
#define FT_TT_UDRV_MRRS_BYTES_DEFAULT 4096u

/** What a set attempt found, and what it did about it. */
typedef struct ft_tt_udrv_max_read_request_report {
    unsigned capability_offset; /**< where the PCI Express capability was found */
    uint16_t devctl_before;
    uint16_t devctl_after;
    unsigned code_before;      /**< DEVCTL bits 14:12 as read */
    unsigned code_after;
    unsigned bytes_before;     /**< 128 << code_before */
    unsigned bytes_after;
    bool already_correct;      /**< it held the wanted size; nothing was written */
    bool written;              /**< a write was issued AND read back as intended */
    const char *failure_stage; /**< NULL on success */
    int failure_errno;
} ft_tt_udrv_max_read_request_report;

/** Bytes for a DEVCTL MRRS code: `128 << code`. Codes above 5 are reserved and
 *  answer 0 rather than a plausible size. */
unsigned ft_tt_udrv_max_read_request_bytes(unsigned code);

/**
 * Read the card's current Max_Read_Request_Size.
 *
 * @param device_dir  directory holding the `config` file
 * @param out_code    DEVCTL bits 14:12
 * @param out_bytes   may be NULL; `128 << code`
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_ENODEV with no PCI Express
 *                    capability; FT_TT_UDRV_EIO when `config` will not answer.
 *                    A NON-ROOT caller lands in the EIO case: the kernel
 *                    truncates an unprivileged read of `config` to the 64-byte
 *                    standard header, so the capability POINTER (0x34) reads and
 *                    the capability it points at (>= 0x40) reads short
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_config_max_read_request_read(const char *device_dir, unsigned *out_code,
                                            unsigned *out_bytes);

/**
 * Set Max_Read_Request_Size to `bytes`, verifying the readback.
 *
 * A device already holding that size is reported `already_correct` and is NOT
 * written — so a success from this function does not mean "the write worked"
 * unless `written` is true. That distinction is the whole gate: on this host the
 * card arrives at 4096 already, and a check that could not tell the two apart
 * would pass on a build with the write deleted.
 *
 * @param device_dir  directory holding the `config` file
 * @param bytes       128, 256, 512, 1024, 2048 or 4096
 * @param out         zeroed and filled; may not be NULL
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL for a size that is not a
 *                    legal MRRS; FT_TT_UDRV_ENODEV with no PCI Express
 *                    capability; FT_TT_UDRV_EIO when the write is refused
 *                    (EACCES for a non-root caller — a config-space write is
 *                    privileged and no udev rule changes that) or when the
 *                    readback disagrees with what was written
 *
 * Thread-safe: no (read-modify-write of one register)   Signal-safe: no
 */
int ft_tt_udrv_config_max_read_request_set(const char *device_dir, unsigned bytes,
                                           ft_tt_udrv_max_read_request_report *out);

/** One line: what MRRS was, what it is, and whether we wrote it. */
void ft_tt_udrv_max_read_request_print(const ft_tt_udrv_max_read_request_report *report);

/* ------------------------------------------------------------------------ *
 * Memory decode — the step that turns the BARs on
 * ------------------------------------------------------------------------ */

/** What `ft_tt_udrv_config_memory_decode_enable()` found and what it did. */
typedef struct {
    uint16_t command_before;   /**< PCI_COMMAND (config 0x04) as read */
    uint16_t command_after;    /**< after the write, or the word that was READ on a failure */
    bool already_enabled;      /**< bit 1 was already set; nothing was written */
    bool written;              /**< a write was issued AND read back with bit 1 set */
    const char *failure_stage; /**< NULL on success */
    int failure_errno;
} ft_tt_udrv_memory_decode_report;

/**
 * Set PCI_COMMAND's Memory Space Enable (bit 1), so BAR0 answers reads.
 *
 * THIS IS THE STEP A COLD HOST HAS NOBODY TO PERFORM, and it is a different
 * question from MRRS even though it fails the same way. `pci_enable_device()`
 * inside the vendor probe is what normally sets this bit, and it SURVIVES an
 * `rmmod` — `pci_disable_device()` leaves memory decode on — so every
 * measurement ever taken on a machine that had once loaded tt-kmd found it
 * already set (the p150a read PCI_COMMAND 0x0102 straight after `rmmod`). On a
 * board that has only ever been powered on, PCI_COMMAND reads 0x0100 (measured
 * on a second card host, 2026-08-18): the BAR mmap SUCCEEDS and every word of it reads
 * 0xFFFFFFFF, which is indistinguishable from dead silicon.
 *
 * Read-modify-write, because PCI_COMMAND also carries Bus Master, SERR and the
 * parity-error response, and a bare store would clear them. Bus Master is
 * DELIBERATELY NOT SET: nothing on this data path has the card read host
 * memory, and granting DMA that no code uses is a privilege we can decline.
 *
 * BEST EFFORT for the same reason MRRS is: a config-space write is privileged,
 * so a non-root run cannot do it and must inherit a card someone else enabled —
 * the udev rule's `setpci COMMAND=0002:0002` line is that someone.
 *
 * @param device_dir  directory holding the `config` file
 * @param out         zeroed and filled; may not be NULL
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL for a NULL report;
 *                    FT_TT_UDRV_EIO when `config` will not answer, when the
 *                    write is refused (EACCES for a non-root caller), or when
 *                    the readback comes back with bit 1 still clear
 *
 * Thread-safe: no (read-modify-write of one register)   Signal-safe: no
 */
int ft_tt_udrv_config_memory_decode_enable(const char *device_dir,
                                           ft_tt_udrv_memory_decode_report *out);

/* ------------------------------------------------------------------------ *
 * Snapshot and restore — the replacement for pci_save_state/pci_restore_state
 * ------------------------------------------------------------------------ */

/**
 * Take everything an in-place reset will destroy. Call this BEFORE setting the reset
 * marker, so the saved command register has the marker clear.
 *
 * @param device_dir  directory holding the `config` file
 * @param out         zeroed and filled; `valid` is true only on FT_TT_UDRV_OK
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_ENODEV if there is no PCI Express capability
 *                    (MRRS could not be restored, so this is a refusal); FT_TT_UDRV_EIO
 */
int ft_tt_udrv_config_snapshot_take(const char *device_dir, ft_tt_udrv_config_snapshot *out);

/**
 * Write a snapshot back, in the kernel's order, and verify every register read back as
 * intended.
 *
 * A register that already holds the wanted value is recorded as `skipped` and not written —
 * the same thing `pci_restore_config_space_range()` does, and it keeps the journal honest
 * about what the reset actually cleared.
 *
 * @param device_dir  directory holding the `config` file
 * @param snapshot    a `valid` snapshot
 * @param out         zeroed and filled; `mismatches == 0` is the success condition
 * @return            FT_TT_UDRV_OK when every write landed and read back; FT_TT_UDRV_EIO
 *                    otherwise, with `failure_stage` naming the step
 */
int ft_tt_udrv_config_snapshot_restore(const char *device_dir,
                                       const ft_tt_udrv_config_snapshot *snapshot,
                                       ft_tt_udrv_config_restore_report *out);

/** One line per step: what was written where, and the verdict. */
void ft_tt_udrv_config_restore_print(const ft_tt_udrv_config_restore_report *report);

/**
 * Which bits of standard-header dword `index` are expected to read back as written.
 *
 * Dword 1 carries the RW1C STATUS half at 0x06: writing a saved value there clears whatever
 * the device has latched since, so the register never reads back as written and an exact
 * comparison would report a phantom mismatch on every restore. Everything else in the first
 * 64 bytes is either writable or read-only, and both compare exactly.
 *
 * The mask also decides whether a register is written at all — a dword whose MASKED bits
 * already match is recorded as `skipped`. Exposed because it is a pure function and the only
 * part of the masking that a fixture can observe: a plain file always reads back exactly what
 * was written to it, so the RW1C behaviour itself is only visible on real silicon.
 */
uint32_t ft_tt_udrv_config_header_verify_mask(unsigned index);

/* ------------------------------------------------------------------------ *
 * The resets themselves
 * ------------------------------------------------------------------------ */

/**
 * ASIC reset — set the reset marker, then arm and fire the interface timer. Three config
 * writes; the card resets itself a moment later and the caller must then wait.
 *
 * DESTRUCTIVE AND CARD-WIDE. Every Tensix returns to power-on default. The caller owns the
 * card lock, has closed every device handle, and has munmap'd every aperture — none of which
 * this function can check, which is why it is not the entry point a command line reaches.
 *
 * @param device_dir  directory holding the `config` file
 * @param out         zeroed and filled
 * @return            FT_TT_UDRV_OK once all three writes landed; FT_TT_UDRV_EIO with
 *                    `failure_stage` naming which one did not
 */
int ft_tt_udrv_reset_asic(const char *device_dir, ft_tt_udrv_reset_report *out);

/**
 * CONFIG_WRITE reset — the interface timer alone, with no reset marker. tt-kmd's flag 2.
 * Weaker than an ASIC reset and kept because it is the one rung between "link" and "ASIC".
 */
int ft_tt_udrv_reset_config_write(const char *device_dir, ft_tt_udrv_reset_report *out);

/**
 * Poll until the card answers config reads with its own vendor id AND has cleared the reset
 * marker it was given.
 *
 * BOTH conditions matter and neither is sufficient. A vendor id of 0xFFFF means config reads
 * are not being answered at all; a still-set marker means the ASIC has not come out of reset
 * yet, and restoring config at that moment is exactly the bug that made tt-kmd's in-place
 * path look impossible (see @status).
 *
 * @param device_dir  directory holding the `config` file
 * @param expect_vendor  the vendor id that means the device is answering
 * @param timeout_ms  give up after this long
 * @param out         zeroed and filled
 * @return            FT_TT_UDRV_OK when both conditions held; FT_TT_UDRV_EAGAIN on timeout
 */
int ft_tt_udrv_reset_wait_ready(const char *device_dir, uint16_t expect_vendor,
                                unsigned timeout_ms, ft_tt_udrv_reset_wait_report *out);

/** One line: what the wait saw. */
void ft_tt_udrv_reset_print(const ft_tt_udrv_reset_report *reset,
                            const ft_tt_udrv_reset_wait_report *wait);

/* ------------------------------------------------------------------------ *
 * The link
 * ------------------------------------------------------------------------ */

/**
 * Read one device's negotiated link state out of its PCI Express capability.
 *
 * @param device_dir  the CARD's or the BRIDGE's sysfs directory — both have a LNKSTA
 * @param out         zeroed and filled
 */
int ft_tt_udrv_link_status_read(const char *device_dir, ft_tt_udrv_link_status *out);

/**
 * Ask a link to retrain, and wait for training to finish.
 *
 * Issue this on the UPSTREAM BRIDGE, not the card: the bridge's LNKCTL is the one
 * `setpci -s $BRIDGE CAP_EXP+10.w=0020:0020` writes, and it is the downstream port that
 * drives the negotiation.
 *
 * @param bridge_dir  the bridge's sysfs directory
 * @param timeout_ms  how long to wait for the Link Training bit to clear
 * @param out         zeroed and filled; `after` is the state it settled at
 * @return            FT_TT_UDRV_OK once training completed; FT_TT_UDRV_EAGAIN if it did not
 *                    clear in time; FT_TT_UDRV_EIO on a config-space failure
 */
int ft_tt_udrv_link_retrain(const char *bridge_dir, unsigned timeout_ms,
                            ft_tt_udrv_link_retrain_report *out);

/** "2.5 GT/s" / "5 GT/s" / "8 GT/s" / ... Never NULL. */
const char *ft_tt_udrv_link_speed_string(unsigned speed_code);

/** Two lines: the link before and after a retrain. */
void ft_tt_udrv_link_retrain_print(const ft_tt_udrv_link_retrain_report *report);

#endif /* FT_TT_UDRV_RESET_H */
