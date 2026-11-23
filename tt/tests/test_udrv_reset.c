/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_reset.c
 * @brief Off-hardware gate for `ft_tt_udrv_reset.[ch]` — the userspace ASIC reset, the PCI
 *   config snapshot/restore that replaces `pci_save_state`/`pci_restore_state`, and the PCIe
 *   link retrain.
 *
 *   Every function in that module reaches hardware through exactly one thing: `pread`/
 *   `pwrite` on `<device_dir>/config`. So a directory holding a synthetic 4096-byte `config`
 *   file exercises all of it, INCLUDING the cases a real card cannot be asked to produce —
 *   a capability list that loops, a version-1 PCI Express capability, a device that answers
 *   0xFFFF, a config file that refuses writes.
 *
 *   WHAT A FIXTURE CANNOT TEST, stated so nobody assumes otherwise: that writing 0x934/0x930
 *   makes the ASIC reset; that the card clears the reset marker on its way back; that the
 *   restored BARs decode. Those are `tests/test_udrv_card_reset.c`'s job, on the card, and
 *   the whole point of the DELIBERATELY WEDGED CORE arm.
 *
 *   THE ORDER OF THE WRITES IS THE LOAD-BEARING PROPERTY and it is checked from the report
 *   JOURNALS, not from the resulting file — config space ends up identical whatever order it
 *   was written in, so a test that only reads the file back is vacuous about ordering. That
 *   is the shape of vacuous gate this project has been caught by four times.
 *
 * @status None
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ft_tt_udrv.h"
#include "ft_tt_udrv_reset.h"

#define CONFIG_BYTES 4096u

/* A fixture's PCI Express capability sits here, after a power-management capability, so the
 * walk has to take a real step rather than finding it first. */
#define FIXTURE_PM_CAP_OFFSET 0x40u
#define FIXTURE_PCIE_CAP_OFFSET 0x80u
#define FIXTURE_VENDOR 0x1e52u
#define FIXTURE_DEVICE 0xb140u

static unsigned checks_run;
static unsigned checks_failed;

static void check(bool condition, const char *what)
{
    checks_run++;
    if (!condition) {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_equal_u32(uint32_t got, uint32_t want, const char *what)
{
    checks_run++;
    if (got != want) {
        checks_failed++;
        printf("  FAIL  %s: got 0x%08x want 0x%08x\n", what, got, want);
    }
}

/* ------------------------------------------------------------------------ *
 * The fixture
 * ------------------------------------------------------------------------ */

typedef struct fixture {
    char directory[128];
    char config_path[160];
    unsigned char bytes[CONFIG_BYTES];
} fixture;

static void fixture_put_dword(fixture *fx, unsigned offset, uint32_t value)
{
    fx->bytes[offset + 0u] = (unsigned char)(value & 0xFFu);
    fx->bytes[offset + 1u] = (unsigned char)((value >> 8u) & 0xFFu);
    fx->bytes[offset + 2u] = (unsigned char)((value >> 16u) & 0xFFu);
    fx->bytes[offset + 3u] = (unsigned char)((value >> 24u) & 0xFFu);
}

static void fixture_put_word(fixture *fx, unsigned offset, uint16_t value)
{
    fx->bytes[offset + 0u] = (unsigned char)(value & 0xFFu);
    fx->bytes[offset + 1u] = (unsigned char)((value >> 8u) & 0xFFu);
}

static uint32_t fixture_get_dword(const fixture *fx, unsigned offset)
{
    return (uint32_t)fx->bytes[offset] | ((uint32_t)fx->bytes[offset + 1u] << 8u) |
           ((uint32_t)fx->bytes[offset + 2u] << 16u) |
           ((uint32_t)fx->bytes[offset + 3u] << 24u);
}

/** Write the in-memory image out to the fixture's `config` file. */
static bool fixture_flush(const fixture *fx)
{
    int fd = open(fx->config_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    ssize_t put;

    if (fd < 0)
        return false;
    put = write(fd, fx->bytes, sizeof(fx->bytes));
    close(fd);
    return put == (ssize_t)sizeof(fx->bytes);
}

/** Read the fixture's `config` file back into the in-memory image. */
static bool fixture_reload(fixture *fx)
{
    int fd = open(fx->config_path, O_RDONLY);
    ssize_t got;

    if (fd < 0)
        return false;
    got = read(fd, fx->bytes, sizeof(fx->bytes));
    close(fd);
    return got == (ssize_t)sizeof(fx->bytes);
}

/**
 * Build a plausible Blackhole-shaped config space: identity words, a command register with
 * memory decode and bus master on, six BARs, a two-entry capability list (power management
 * then PCI Express), and a PCI Express capability at `capability_version`.
 */
static bool fixture_build(fixture *fx, const char *name, unsigned capability_version)
{
    memset(fx, 0, sizeof(*fx));
    snprintf(fx->directory, sizeof(fx->directory), "/tmp/ft_tt_udrv_reset_%s", name);
    snprintf(fx->config_path, sizeof(fx->config_path), "%s/config", fx->directory);
    if (mkdir(fx->directory, 0700) != 0 && errno != EEXIST)
        return false;
    /* A previous run's read-only arm may have left the file at 0400, and O_TRUNC would then
     * fail. Make it writable again rather than depending on how the last run ended. */
    (void)chmod(fx->config_path, 0600);

    fixture_put_dword(fx, 0x00u, ((uint32_t)FIXTURE_DEVICE << 16u) | FIXTURE_VENDOR);
    /* command 0x0006 = memory decode + bus master; status half deliberately nonzero so the
     * dword-1 verify mask has something to ignore. */
    fixture_put_dword(fx, 0x04u, 0x00100006u);
    fixture_put_dword(fx, 0x08u, 0x12000001u); /* revision / class */
    fixture_put_dword(fx, 0x0Cu, 0x00000010u); /* cacheline size */
    fixture_put_dword(fx, 0x10u, 0xC0000004u); /* BAR0, 64-bit memory */
    fixture_put_dword(fx, 0x14u, 0x00000021u);
    fixture_put_dword(fx, 0x18u, 0xD0000004u); /* BAR2 */
    fixture_put_dword(fx, 0x1Cu, 0x00000021u);
    fixture_put_dword(fx, 0x20u, 0x00000000u);
    fixture_put_dword(fx, 0x24u, 0x00000000u);
    fixture_put_dword(fx, 0x28u, 0x00000000u); /* cardbus CIS */
    fixture_put_dword(fx, 0x2Cu, 0xb140abcdu); /* subsystem ids */
    fixture_put_dword(fx, 0x30u, 0x00000000u); /* expansion ROM */
    fixture_put_dword(fx, 0x34u, FIXTURE_PM_CAP_OFFSET);
    fixture_put_dword(fx, 0x38u, 0x00000000u);
    fixture_put_dword(fx, 0x3Cu, 0x0000010bu); /* interrupt line / pin */

    /* capability 0x01 (power management) -> 0x80 */
    fixture_put_dword(fx, FIXTURE_PM_CAP_OFFSET, (FIXTURE_PCIE_CAP_OFFSET << 8u) | 0x01u);
    /* capability 0x10 (PCI Express), end of list */
    /* Bytes 2-3 of a capability dword are PCI_EXP_FLAGS: bits 3:0 the capability VERSION,
     * bits 7:4 the device/port type (0 = endpoint). */
    fixture_put_dword(fx, FIXTURE_PCIE_CAP_OFFSET,
                      ((uint32_t)(capability_version & 0x0Fu) << 16u) |
                          FT_TT_UDRV_CAP_ID_EXPRESS);
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x08u, 0x5000u); /* DEVCTL, MRRS = 4096 */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x10u, 0x0040u); /* LNKCTL */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x12u, 0x0012u); /* LNKSTA: 5 GT/s x1 */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x18u, 0x0000u); /* SLTCTL */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x1Cu, 0x0000u); /* RTCTL */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x28u, 0x0010u); /* DEVCTL2 */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x30u, 0x0007u); /* LNKCTL2 */
    fixture_put_word(fx, FIXTURE_PCIE_CAP_OFFSET + 0x38u, 0x0000u); /* SLTCTL2 */

    return fixture_flush(fx);
}

/** What a card looks like the instant it comes out of an ASIC reset: header cleared. */
static void fixture_simulate_post_reset(fixture *fx)
{
    unsigned index;

    for (index = 1u; index < 16u; index++)
        fixture_put_dword(fx, index * 4u, 0x00000000u);
    /* identity survives; the card answers with its vendor id and a zeroed command register,
     * which is the "marker cleared, memory decode off" state the driver sees. */
    (void)fixture_flush(fx);
}

/* ------------------------------------------------------------------------ *
 * 1. the capability walk
 * ------------------------------------------------------------------------ */

static void stage_capability_walk(void)
{
    fixture fx;
    unsigned offset = 0u;

    printf("1. capability walk\n");
    if (!fixture_build(&fx, "caps", 2u)) {
        printf("  FAIL  could not build the fixture\n");
        checks_run++;
        checks_failed++;
        return;
    }
    check(ft_tt_udrv_config_find_capability(fx.directory, FT_TT_UDRV_CAP_ID_EXPRESS, &offset) ==
              FT_TT_UDRV_OK,
          "the PCI Express capability is found");
    check_equal_u32(offset, FIXTURE_PCIE_CAP_OFFSET, "at the fixture's offset");

    /* A well-formed list that simply lacks the capability is ENODEV, not EIO. */
    fixture_put_dword(&fx, FIXTURE_PCIE_CAP_OFFSET, 0x00000005u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_find_capability(fx.directory, FT_TT_UDRV_CAP_ID_EXPRESS, &offset) ==
              FT_TT_UDRV_ENODEV,
          "a list without the capability is ENODEV");

    /* A LOOP must be refused, and the refusal must RETURN. An unbounded walk hangs here. */
    fixture_put_dword(&fx, FIXTURE_PM_CAP_OFFSET, (FIXTURE_PM_CAP_OFFSET << 8u) | 0x01u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_find_capability(fx.directory, FT_TT_UDRV_CAP_ID_EXPRESS, &offset) ==
              FT_TT_UDRV_EIO,
          "a self-referencing capability list is refused");

    /* A pointer into the standard header is malformed, not a capability. */
    fixture_put_dword(&fx, 0x34u, 0x10u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_find_capability(fx.directory, FT_TT_UDRV_CAP_ID_EXPRESS, &offset) ==
              FT_TT_UDRV_EIO,
          "a capability pointer inside the header is refused");

    /* An all-ones config space -- a device that is not answering -- must not be walked. */
    memset(fx.bytes, 0xFF, sizeof(fx.bytes));
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_find_capability(fx.directory, FT_TT_UDRV_CAP_ID_EXPRESS, &offset) ==
              FT_TT_UDRV_EIO,
          "an all-ones config space is refused, not walked");
}

/* ------------------------------------------------------------------------ *
 * 2. the snapshot
 * ------------------------------------------------------------------------ */

static void stage_snapshot(void)
{
    ft_tt_udrv_config_snapshot snapshot;
    fixture fx;
    unsigned index;

    printf("2. snapshot\n");
    if (!fixture_build(&fx, "snap", 2u))
        return;

    check(ft_tt_udrv_config_snapshot_take(fx.directory, &snapshot) == FT_TT_UDRV_OK,
          "a version-2 capability snapshots");
    check(snapshot.valid, "the snapshot is marked valid");
    check_equal_u32(snapshot.vendor_id, FIXTURE_VENDOR, "vendor id");
    check_equal_u32(snapshot.device_id, FIXTURE_DEVICE, "device id");
    check_equal_u32(snapshot.pcie_capability_offset, FIXTURE_PCIE_CAP_OFFSET, "capability offset");
    check_equal_u32(snapshot.pcie_capability_version, 2u, "capability version");

    for (index = 0u; index < FT_TT_UDRV_CONFIG_HEADER_DWORDS; index++)
        check_equal_u32(snapshot.header[index], fixture_get_dword(&fx, index * 4u),
                        "a header dword is captured verbatim");
    for (index = 0u; index < FT_TT_UDRV_PCIE_SAVED_WORDS; index++)
        check(snapshot.pcie_word_present[index], "all seven capability words are present at v2");
    check_equal_u32(snapshot.pcie_word[0], 0x5000u, "DEVCTL, which carries MRRS");
    check_equal_u32(snapshot.pcie_word[5], 0x0007u, "LNKCTL2");

    /* A VERSION-1 capability has no DEVCTL2/LNKCTL2/SLTCTL2. Reading those offsets would
     * read whatever follows the capability, and WRITING them would corrupt it. */
    if (!fixture_build(&fx, "snapv1", 1u))
        return;
    check(ft_tt_udrv_config_snapshot_take(fx.directory, &snapshot) == FT_TT_UDRV_OK,
          "a version-1 capability snapshots");
    check_equal_u32(snapshot.pcie_capability_version, 1u, "version 1 is reported");
    check(snapshot.pcie_word_present[0] && snapshot.pcie_word_present[3],
          "the four version-1 words are present");
    check(!snapshot.pcie_word_present[4] && !snapshot.pcie_word_present[5] &&
              !snapshot.pcie_word_present[6],
          "the three version-2 words are ABSENT at version 1");

    /* No PCI Express capability at all is a refusal: MRRS could not be restored. */
    if (!fixture_build(&fx, "snapnocap", 2u))
        return;
    fixture_put_dword(&fx, FIXTURE_PCIE_CAP_OFFSET, 0x00000005u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_snapshot_take(fx.directory, &snapshot) == FT_TT_UDRV_ENODEV,
          "a device with no PCI Express capability is refused");
    check(!snapshot.valid, "and the snapshot is not marked valid");
}

/* ------------------------------------------------------------------------ *
 * 3. the restore, and its ORDER
 * ------------------------------------------------------------------------ */

/** Where in the journal a given config offset was written, or -1. */
static int journal_position(const ft_tt_udrv_config_restore_report *report, unsigned offset)
{
    unsigned index;

    for (index = 0u; index < report->step_count; index++)
        if (report->step[index].offset == offset)
            return (int)index;
    return -1;
}

static void stage_restore_order(void)
{
    ft_tt_udrv_config_restore_report report;
    ft_tt_udrv_config_snapshot snapshot;
    fixture fx;
    int command_at;
    int index;

    printf("3. restore order\n");
    if (!fixture_build(&fx, "order", 2u))
        return;
    if (ft_tt_udrv_config_snapshot_take(fx.directory, &snapshot) != FT_TT_UDRV_OK) {
        printf("  FAIL  snapshot\n");
        checks_run++;
        checks_failed++;
        return;
    }
    fixture_simulate_post_reset(&fx);

    check(ft_tt_udrv_config_snapshot_restore(fx.directory, &snapshot, &report) == FT_TT_UDRV_OK,
          "the restore succeeds against a cleared header");
    check_equal_u32(report.mismatches, 0u, "no register failed to read back");
    check_equal_u32(report.step_count, 23u, "7 capability words + 16 header dwords");

    /* The seven capability words come FIRST -- pci_restore_state() does the capability before
     * pci_restore_config_space(). */
    for (index = 0; index < 7; index++)
        check(report.step[index].width == 2u &&
                  report.step[index].offset >= FIXTURE_PCIE_CAP_OFFSET,
              "the first seven steps are capability words");

    /* THE PROPERTY THAT MATTERS: every BAR dword is written BEFORE the command register.
     * Restoring COMMAND first enables a decoder aimed at whatever the BARs hold, which after
     * a reset is address zero. */
    command_at = journal_position(&report, 0x04u);
    check(command_at >= 0, "the command register is in the journal");
    for (index = 0x10; index <= 0x24; index += 4) {
        int bar_at = journal_position(&report, (unsigned)index);

        check(bar_at >= 0 && command_at >= 0 && bar_at < command_at,
              "a BAR is restored before the command register");
    }
    /* Dwords 10-15 precede the BARs, matching pci_restore_config_space()'s three ranges. */
    check(journal_position(&report, 0x28u) < journal_position(&report, 0x10u),
          "header dwords 10-15 precede the BARs");
    check(journal_position(&report, 0x3Cu) < journal_position(&report, 0x10u),
          "the last of that range still precedes the BARs");

    /* And the card really does hold the snapshot again. */
    check(fixture_reload(&fx), "the fixture reloads");
    for (index = 0; index < 16; index++) {
        if (index == 1)
            continue; /* the status half is RW1C and is masked out of the comparison */
        check_equal_u32(fixture_get_dword(&fx, (unsigned)index * 4u),
                        snapshot.header[index], "a header dword is back");
    }
    check_equal_u32(fixture_get_dword(&fx, 0x04u) & 0xFFFFu, snapshot.header[1] & 0xFFFFu,
                    "the command half of dword 1 is back");
}

static void stage_restore_behaviour(void)
{
    ft_tt_udrv_config_restore_report report;
    ft_tt_udrv_config_snapshot snapshot;
    fixture fx;

    printf("4. restore behaviour\n");
    if (!fixture_build(&fx, "behave", 2u))
        return;
    if (ft_tt_udrv_config_snapshot_take(fx.directory, &snapshot) != FT_TT_UDRV_OK)
        return;

    /* Restoring onto an UNCHANGED device writes nothing: every register already holds it.
     * That is also what says the skip logic is not silently skipping everything above. */
    check(ft_tt_udrv_config_snapshot_restore(fx.directory, &snapshot, &report) == FT_TT_UDRV_OK,
          "restoring an unchanged device succeeds");
    check_equal_u32(report.written, 0u, "and writes nothing at all");
    check_equal_u32(report.step_count, 23u, "while still journalling every step");

    /* One changed BAR: exactly one write. */
    fixture_put_dword(&fx, 0x10u, 0x00000004u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_snapshot_restore(fx.directory, &snapshot, &report) == FT_TT_UDRV_OK,
          "one changed BAR restores");
    check_equal_u32(report.written, 1u, "and costs exactly one write");
    check(journal_position(&report, 0x10u) >= 0 &&
              !report.step[journal_position(&report, 0x10u)].skipped,
          "the BAR's step is not marked skipped");

    /* The RW1C status half of dword 1 is masked out of both the compare and the verify. A
     * device that latched a new status bit during the reset must not make the restore write
     * (and then report a phantom mismatch on) a register whose writable half is already
     * right. `written == 0` is the observable: without the mask this arm writes dword 1.
     *
     * NOTE what a fixture cannot show: that the register does not read back as written. A
     * plain file always does. The mask's own value is checked directly, below. */
    check(fixture_reload(&fx), "the fixture reloads before the status-half arm");
    fixture_put_dword(&fx, 0x04u, 0x02900006u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_config_snapshot_restore(fx.directory, &snapshot, &report) == FT_TT_UDRV_OK,
          "a changed status half does not fail the restore");
    check_equal_u32(report.mismatches, 0u, "and is not counted as a mismatch");
    check_equal_u32(report.written, 0u, "and dword 1 is not rewritten for it");
    check_equal_u32(ft_tt_udrv_config_header_verify_mask(1u), 0x0000FFFFu,
                    "dword 1's verify mask covers the command half only");
    check_equal_u32(ft_tt_udrv_config_header_verify_mask(0u), 0xFFFFFFFFu,
                    "every other header dword is compared whole");
    check_equal_u32(ft_tt_udrv_config_header_verify_mask(4u), 0xFFFFFFFFu,
                    "including the BARs");

    /* An invalid snapshot is refused rather than writing zeroes over the header. */
    memset(&snapshot, 0, sizeof(snapshot));
    check(ft_tt_udrv_config_snapshot_restore(fx.directory, &snapshot, &report) ==
              FT_TT_UDRV_EINVAL,
          "an invalid snapshot is refused");
    check(report.step_count == 0u, "and nothing was written");
}

/* ------------------------------------------------------------------------ *
 * 5. the resets
 * ------------------------------------------------------------------------ */

static void stage_reset_writes(void)
{
    ft_tt_udrv_reset_report report;
    fixture fx;

    printf("5. the reset writes\n");
    if (!fixture_build(&fx, "reset", 2u))
        return;

    check(ft_tt_udrv_reset_asic(fx.directory, &report) == FT_TT_UDRV_OK, "the ASIC reset runs");
    check(report.marker_set && report.timer_target_written && report.timer_control_written,
          "all three writes are reported");
    check_equal_u32(report.step_count, 3u, "the journal holds exactly three writes");

    /* ORDER, read from the journal: marker, then timer TARGET, then timer CONTROL. */
    check_equal_u32(report.step[0].offset, 0x004u, "step 0 is PCI_COMMAND (the reset marker)");
    check_equal_u32(report.step[0].width, 2u, "and it is a 16-bit write");
    check_equal_u32(report.step[1].offset, 0x934u, "step 1 arms the interface timer target");
    check_equal_u32(report.step[2].offset, 0x930u, "step 2 fires the interface timer");
    check_equal_u32(report.step[1].value, 0x00000001u, "0x934 <- 1");
    check_equal_u32(report.step[2].value, 0x00000011u, "0x930 <- 0x11 (EN | FORCE_PENDING)");
    check_equal_u32(report.step[0].value & FT_TT_UDRV_COMMAND_PARITY, FT_TT_UDRV_COMMAND_PARITY,
                    "the marker bit is set in the value written");
    check_equal_u32(report.step[0].value & 0x0006u, 0x0006u,
                    "and the rest of the command register is preserved");

    check(fixture_reload(&fx), "the fixture reloads");
    check_equal_u32(fixture_get_dword(&fx, 0x004u) & 0xFFFFu, 0x0046u, "config 0x04 on the card");
    check_equal_u32(fixture_get_dword(&fx, 0x934u), 0x00000001u, "config 0x934 on the card");
    check_equal_u32(fixture_get_dword(&fx, 0x930u), 0x00000011u, "config 0x930 on the card");

    /* CONFIG_WRITE is the same two timer writes with NO marker. */
    if (!fixture_build(&fx, "cfgwrite", 2u))
        return;
    check(ft_tt_udrv_reset_config_write(fx.directory, &report) == FT_TT_UDRV_OK,
          "the CONFIG_WRITE reset runs");
    check(!report.marker_set, "it does not set the reset marker");
    check_equal_u32(report.step_count, 2u, "and issues only the two timer writes");
    check_equal_u32(report.step[0].offset, 0x934u, "target first");
    check_equal_u32(report.step[1].offset, 0x930u, "control second");
    check(fixture_reload(&fx), "the fixture reloads");
    check_equal_u32(fixture_get_dword(&fx, 0x004u) & 0xFFFFu, 0x0006u,
                    "the command register is untouched");

    /* A CONFIG SPACE THAT REFUSES WRITES must stop the sequence at the marker and never fire
     * the timer. This is what proves the marker precedes the timer even if the journal were
     * wrong: a card that cannot take the marker must not be reset. */
    if (!fixture_build(&fx, "readonly", 2u))
        return;
    if (chmod(fx.config_path, 0400) == 0) {
        check(ft_tt_udrv_reset_asic(fx.directory, &report) != FT_TT_UDRV_OK,
              "a read-only config space fails the reset");
        check(!report.marker_set, "the marker did not land");
        check(!report.timer_target_written && !report.timer_control_written,
              "and NEITHER timer write was issued");
        check(report.failure_stage != NULL, "the failure names its stage");
        (void)chmod(fx.config_path, 0600);
    }
}

/* ------------------------------------------------------------------------ *
 * 6. the wait
 * ------------------------------------------------------------------------ */

static void stage_wait_ready(void)
{
    ft_tt_udrv_reset_wait_report report;
    fixture fx;

    printf("6. waiting for the card to come back\n");
    if (!fixture_build(&fx, "wait", 2u))
        return;

    /* Vendor id present, marker clear: ready immediately. */
    check(ft_tt_udrv_reset_wait_ready(fx.directory, FIXTURE_VENDOR, 200u, &report) ==
              FT_TT_UDRV_OK,
          "a healthy device is ready at once");
    check(report.vendor_id_returned && report.marker_cleared, "both conditions are reported met");
    check(!report.timed_out, "and it did not time out");

    /* MARKER STILL SET: not ready. This is the exact state in which restoring config loses
     * the restore in tt-kmd, so it must not be reported ready. */
    fixture_put_dword(&fx, 0x04u, 0x00100046u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_reset_wait_ready(fx.directory, FIXTURE_VENDOR, 150u, &report) ==
              FT_TT_UDRV_EAGAIN,
          "a still-set reset marker is NOT ready");
    check(report.vendor_id_returned, "even though config reads answer");
    check(!report.marker_cleared, "the marker is reported still set");
    check(report.timed_out, "and the wait times out");

    /* A DEVICE ANSWERING 0xFFFF is not ready either, marker or no marker. */
    memset(fx.bytes, 0xFF, sizeof(fx.bytes));
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_reset_wait_ready(fx.directory, FIXTURE_VENDOR, 150u, &report) ==
              FT_TT_UDRV_EAGAIN,
          "an all-ones device is NOT ready");
    check(!report.vendor_id_returned, "the vendor id is not the card's");
    check(!report.marker_cleared,
          "and 0xFFFF is not read as a cleared marker just because the bit is set");
    check(report.polls >= 2u, "the wait actually polled more than once");
}

/* ------------------------------------------------------------------------ *
 * 7. the link
 * ------------------------------------------------------------------------ */

static void stage_link(void)
{
    ft_tt_udrv_link_retrain_report retrain;
    ft_tt_udrv_link_status status;
    fixture fx;

    printf("7. the link\n");
    if (!fixture_build(&fx, "link", 2u))
        return;

    check(ft_tt_udrv_link_status_read(fx.directory, &status) == FT_TT_UDRV_OK, "LNKSTA reads");
    check_equal_u32(status.speed_code, FT_TT_UDRV_LINK_SPEED_5GT, "5 GT/s is decoded");
    check_equal_u32(status.width, 1u, "x1 is decoded");
    check(!status.training, "and the link is not training");
    check(strcmp(ft_tt_udrv_link_speed_string(FT_TT_UDRV_LINK_SPEED_2_5GT), "2.5 GT/s") == 0,
          "2.5 GT/s stringifies");
    check(strcmp(ft_tt_udrv_link_speed_string(FT_TT_UDRV_LINK_SPEED_5GT), "5 GT/s") == 0,
          "5 GT/s stringifies");

    /* A Gen1 link, the state every reset leaves behind. */
    fixture_put_word(&fx, FIXTURE_PCIE_CAP_OFFSET + 0x12u, 0x0011u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_link_status_read(fx.directory, &status) == FT_TT_UDRV_OK,
          "a Gen1 LNKSTA reads");
    check_equal_u32(status.speed_code, FT_TT_UDRV_LINK_SPEED_2_5GT, "2.5 GT/s is decoded");

    /* The retrain writes LNKCTL with bit 5 set and leaves the rest alone. The fixture never
     * clears the training bit by itself, so training-not-complete is the honest answer. */
    check(ft_tt_udrv_link_retrain(fx.directory, 60u, &retrain) == FT_TT_UDRV_OK,
          "a link that is not training completes at once");
    check(retrain.retrain_requested, "the retrain was requested");
    check(fixture_reload(&fx), "the fixture reloads");
    check_equal_u32(fixture_get_dword(&fx, FIXTURE_PCIE_CAP_OFFSET + 0x10u) & 0xFFFFu, 0x0060u,
                    "LNKCTL got bit 5 set with 0x0040 preserved");

    /* A link stuck in training must time out and SAY so, not report success. */
    fixture_put_word(&fx, FIXTURE_PCIE_CAP_OFFSET + 0x12u,
                     (uint16_t)(0x0011u | FT_TT_UDRV_LNKSTA_TRAINING));
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_link_retrain(fx.directory, 60u, &retrain) == FT_TT_UDRV_EAGAIN,
          "a link stuck in training times out");
    check(!retrain.training_completed, "and does not claim training completed");
    check(retrain.after.training, "the report says it is still training");

    /* A device with no PCI Express capability cannot have its link read. */
    if (!fixture_build(&fx, "linknocap", 2u))
        return;
    fixture_put_dword(&fx, 0x34u, 0x00000000u);
    (void)fixture_flush(&fx);
    check(ft_tt_udrv_link_status_read(fx.directory, &status) == FT_TT_UDRV_ENODEV,
          "no capability, no link status");
    check(ft_tt_udrv_link_retrain(fx.directory, 60u, &retrain) == FT_TT_UDRV_ENODEV,
          "and no retrain either");
    check(!retrain.retrain_requested, "nothing was written");
}

/* ------------------------------------------------------------------------ *
 * 8. arguments
 * ------------------------------------------------------------------------ */

static void stage_arguments(void)
{
    ft_tt_udrv_config_restore_report restore = {{{0u, 0u, 0u, false}}, 0u, 0u, 0u, 0u, NULL, 0};
    ft_tt_udrv_config_snapshot snapshot;
    ft_tt_udrv_reset_report reset;
    uint32_t value = 0u;

    memset(&snapshot, 0, sizeof(snapshot));
    memset(&reset, 0, sizeof(reset));

    printf("8. arguments and absent devices\n");
    check(ft_tt_udrv_config_read_dword(NULL, 0u, &value) == FT_TT_UDRV_EINVAL,
          "a NULL device directory is refused");
    check(ft_tt_udrv_config_read_dword("/tmp/ft_tt_udrv_reset_nope", 0u, &value) ==
              FT_TT_UDRV_EIO,
          "a directory with no config file is an I/O failure");
    check(ft_tt_udrv_config_read_dword("/tmp", 2u, &value) == FT_TT_UDRV_EINVAL,
          "a misaligned dword offset is refused");
    check(ft_tt_udrv_config_read_word("/tmp", 1u, (uint16_t *)NULL) == FT_TT_UDRV_EINVAL,
          "a NULL output word is refused");
    check(ft_tt_udrv_config_find_capability("/tmp", FT_TT_UDRV_CAP_ID_EXPRESS, NULL) ==
              FT_TT_UDRV_EINVAL,
          "a NULL capability output is refused");
    check(ft_tt_udrv_config_snapshot_take("/tmp", NULL) == FT_TT_UDRV_EINVAL,
          "a NULL snapshot is refused");
    check(ft_tt_udrv_config_snapshot_restore("/tmp", NULL, &restore) == FT_TT_UDRV_EINVAL,
          "a NULL snapshot cannot be restored");
    check(ft_tt_udrv_config_snapshot_restore("/tmp", &snapshot, NULL) == FT_TT_UDRV_EINVAL,
          "a NULL restore report is refused");
    check(ft_tt_udrv_reset_asic("/tmp", NULL) == FT_TT_UDRV_EINVAL,
          "a NULL reset report is refused");
    check(ft_tt_udrv_reset_asic("/tmp/ft_tt_udrv_reset_nope", &reset) == FT_TT_UDRV_EIO,
          "an absent device cannot be reset");
    check(reset.step_count == 0u, "and nothing was journalled");
    (void)restore;
}

int main(void)
{
    printf("test_udrv_reset — the userspace reset path, off hardware\n\n");

    stage_capability_walk();
    stage_snapshot();
    stage_restore_order();
    stage_restore_behaviour();
    stage_reset_writes();
    stage_wait_ready();
    stage_link();
    stage_arguments();

    printf("\n%u checks, %u failures\n", checks_run, checks_failed);
    return checks_failed == 0u ? 0 : 1;
}
