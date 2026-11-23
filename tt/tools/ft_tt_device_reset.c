/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_device_reset.c
 * @brief Implementation of the device-level (`RESET_DEVICE`) and bus-level (PCIe
 *   Secondary Bus Reset) recovery rungs.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_device_reset.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_device.h"
#include "ft_tt_kmd_abi.h"

/** PCI configuration header: the command register, and its memory-decode bit. */
#define PCI_COMMAND_OFFSET 0x04u
#define PCI_COMMAND_MEMORY 0x0002u

/** PCI-to-PCI bridge configuration header, offset of the Bridge Control register. */
#define PCI_BRIDGE_CONTROL_OFFSET 0x3Eu

/** Bridge Control bit 6: assert reset on the secondary bus (PCIe PERST# to its children). */
#define PCI_BRIDGE_CONTROL_SECONDARY_BUS_RESET 0x0040u

/** How long reset is held asserted. The PCIe base spec demands at least 1 ms; 100 ms is
 *  what every in-tree implementation of this uses and there is nothing to win by cutting
 *  it on a recovery path. */
#define SBR_ASSERT_MILLISECONDS 100u

/** How long the card is given to re-train and boot its ARC from SPI before the rescan. */
#define SBR_SETTLE_MILLISECONDS 2000u

/** How long the device is given to reappear in sysfs after the rescan. */
#define SBR_REAPPEAR_MILLISECONDS 10000u

#define PCI_CONFIG_SAVE_BYTES 4096u

/** How many times POST_RESET is retried, and how long between attempts. */
#define POST_RESET_ATTEMPTS 10u
#define POST_RESET_INTERVAL_MILLISECONDS 1000u

#define SYSFS_PCI_DEVICES "/sys/bus/pci/devices"

/** Tenstorrent's PCI vendor id. The card must be this or the SBR is refused. */
#define FT_TT_PCI_VENDOR_ID 0x1e52u

/** tt-kmd's `struct tenstorrent_reset_device`. Declared here rather than
 *  in the ioctl ABI header (tt/ft_tt_kmd_abi.h) on purpose: that header carries payload structs only for
 *  the ioctls Foundry issues, so that nothing else CAN be issued by accident. This module
 *  is the one place allowed to issue this one. */
struct ft_tt_reset_device_ioctl {
    struct {
        uint32_t output_size_bytes;
        uint32_t flags;
    } in;
    struct {
        uint32_t output_size_bytes;
        uint32_t result;
    } out;
};

/** Record which sysfs step failed and hand back FT_TT_EIO. */
static int sysfs_step_failed(const char **out_stage, int *out_errno, const char *stage)
{
    *out_stage = stage;
    *out_errno = errno;
    return FT_TT_EIO;
}

static void sleep_milliseconds(unsigned milliseconds)
{
    struct timespec request;

    request.tv_sec = (time_t)(milliseconds / 1000u);
    request.tv_nsec = (long)(milliseconds % 1000u) * 1000000L;
    while (nanosleep(&request, &request) != 0 && errno == EINTR)
        ;
}

/* ================================================================
 * RESET_DEVICE — rungs 3 and 4
 * ================================================================ */

/** Issue one flag on an already-open fd, recording the driver's own result word. */
static int issue_reset_flag(int fd, ft_tt_device_reset_flag flag, uint32_t *out_result,
                            int *out_errno)
{
    struct ft_tt_reset_device_ioctl request;

    memset(&request, 0, sizeof(request));
    request.in.output_size_bytes = (uint32_t)sizeof(request.out);
    request.in.flags = (uint32_t)flag;

    *out_errno = 0;
    if (ioctl(fd, FT_TT_KMD_IOCTL_RESET_DEVICE, &request) != 0) {
        *out_errno = errno;
        *out_result = 0;
        return FT_TT_EIO;
    }
    *out_result = request.out.result;
    return FT_TT_OK;
}

int ft_tt_device_reset_issue(const char *chardev_path, ft_tt_device_reset_flag flag,
                             ft_tt_device_reset_report *out)
{
    struct stat node;
    int fd;

    if (!chardev_path || !out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));
    out->flag = flag;

    fd = open(chardev_path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        ft_tt_log_error("reset: cannot open %s: %s", chardev_path, strerror(errno));
        return FT_TT_EIO;
    }

    (void)issue_reset_flag(fd, flag, &out->driver_result, &out->ioctl_errno);

    /* POST_RESET is the completion half — it restores PCI state and re-runs init_hardware. It has to be RETRIED: its first check is that the reset marker
     * (the PCI parity-error-response bit) has been cleared BY the reset, and a card still
     * in reset answers 0xffff to that read, which looks like the marker is still set.
     * Issued with no delay it returns FAILED on a reset that worked (measured). */
    out->post_reset_issued = true;
    for (out->post_reset_attempts = 1u; out->post_reset_attempts <= POST_RESET_ATTEMPTS;
         out->post_reset_attempts++) {
        sleep_milliseconds(POST_RESET_INTERVAL_MILLISECONDS);
        (void)issue_reset_flag(fd, FT_TT_RESET_DEVICE_POST_RESET, &out->post_reset_driver_result,
                               &out->post_reset_errno);
        if (out->post_reset_errno == 0 && out->post_reset_driver_result == 0u)
            break;
    }
    close(fd);

    out->chardev_present_after = stat(chardev_path, &node) == 0;
    return FT_TT_OK;
}

void ft_tt_device_reset_print(const ft_tt_device_reset_report *report)
{
    if (!report)
        return;
    printf("RESET_DEVICE flag %u: ", (unsigned)report->flag);
    if (report->ioctl_errno != 0)
        printf("ioctl failed, errno %d (%s)\n", report->ioctl_errno,
               strerror(report->ioctl_errno));
    else
        printf("driver result=%u (%s)\n", report->driver_result,
               report->driver_result == 0u ? "success" : "FAILED");
    if (report->post_reset_issued) {
        printf("  POST_RESET (%u attempt%s): ", report->post_reset_attempts,
               report->post_reset_attempts == 1u ? "" : "s");
        if (report->post_reset_errno != 0)
            printf("ioctl failed, errno %d (%s)\n", report->post_reset_errno,
                   strerror(report->post_reset_errno));
        else
            printf("driver result=%u (%s)\n", report->post_reset_driver_result,
                   report->post_reset_driver_result == 0u ? "success" : "FAILED");
    }
    printf("  chardev present afterwards: %s\n", report->chardev_present_after ? "yes" : "no");
}

/* ================================================================
 * sysfs helpers
 * ================================================================ */

/** Read a whole small sysfs file into `buffer` as a NUL-terminated string. */
static int read_sysfs_text(const char *path, char *buffer, size_t bytes)
{
    ssize_t got;
    int fd = open(path, O_RDONLY | O_CLOEXEC);

    if (fd < 0)
        return FT_TT_EIO;
    got = read(fd, buffer, bytes - 1u);
    close(fd);
    if (got < 0)
        return FT_TT_EIO;
    buffer[got] = '\0';
    return FT_TT_OK;
}

/** Write one byte of text to a sysfs control file ("1" to remove, rescan, ...). */
static int write_sysfs_text(const char *path, const char *value)
{
    ssize_t written;
    int fd = open(path, O_WRONLY | O_CLOEXEC);

    if (fd < 0)
        return FT_TT_EIO;
    written = write(fd, value, strlen(value));
    close(fd);
    return written == (ssize_t)strlen(value) ? FT_TT_OK : FT_TT_EIO;
}

/** A 16-bit read of one device's PCI configuration space through sysfs. */
static int read_config_word(const char *bdf, unsigned offset, uint16_t *out_value)
{
    char path[256];
    uint8_t bytes[2];
    ssize_t got;
    int fd;

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s/config", bdf) < 0)
        return FT_TT_EIO;
    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return FT_TT_EIO;
    got = pread(fd, bytes, sizeof(bytes), (off_t)offset);
    close(fd);
    if (got != (ssize_t)sizeof(bytes))
        return FT_TT_EIO;
    *out_value = (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
    return FT_TT_OK;
}

/** A 16-bit write into one device's PCI configuration space through sysfs. Root only. */
static int write_config_word(const char *bdf, unsigned offset, uint16_t value)
{
    char path[256];
    uint8_t bytes[2];
    ssize_t written;
    int fd;

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s/config", bdf) < 0)
        return FT_TT_EIO;
    bytes[0] = (uint8_t)(value & 0xFFu);
    bytes[1] = (uint8_t)((value >> 8) & 0xFFu);
    fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0)
        return FT_TT_EIO;
    written = pwrite(fd, bytes, sizeof(bytes), (off_t)offset);
    close(fd);
    return written == (ssize_t)sizeof(bytes) ? FT_TT_OK : FT_TT_EIO;
}

/** Copy a device's configuration space to a file, for the evidence a reset would erase. */
static int save_config_space(const char *bdf, const char *directory, const char *label)
{
    char source_path[256];
    char destination_path[512];
    uint8_t buffer[PCI_CONFIG_SAVE_BYTES];
    ssize_t got;
    ssize_t written;
    int source;
    int destination;

    if (snprintf(source_path, sizeof(source_path), SYSFS_PCI_DEVICES "/%s/config", bdf) < 0)
        return FT_TT_EIO;
    if (snprintf(destination_path, sizeof(destination_path), "%s/%s_%s_config.bin", directory,
                 label, bdf) < 0)
        return FT_TT_EIO;

    source = open(source_path, O_RDONLY | O_CLOEXEC);
    if (source < 0)
        return FT_TT_EIO;
    got = read(source, buffer, sizeof(buffer));
    close(source);
    if (got <= 0)
        return FT_TT_EIO;

    destination = open(destination_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (destination < 0)
        return FT_TT_EIO;
    written = write(destination, buffer, (size_t)got);
    close(destination);
    return written == got ? FT_TT_OK : FT_TT_EIO;
}

/** Length of BAR0 as sysfs reports it, or 0 if it is unassigned/absent. */
static uint64_t read_bar0_length(const char *bdf)
{
    char path[256];
    char text[512];
    unsigned long long start = 0;
    unsigned long long end = 0;

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s/resource", bdf) < 0)
        return 0;
    if (read_sysfs_text(path, text, sizeof(text)) != FT_TT_OK)
        return 0;
    if (sscanf(text, "%llx %llx", &start, &end) != 2)
        return 0;
    return end > start ? (uint64_t)(end - start + 1ull) : 0u;
}

/** Characters in "dddd:bb:dd.f", without the terminator. */
#define PCI_ADDRESS_LENGTH 12u

/** True when `name` looks like "dddd:bb:dd.f". */
static bool looks_like_pci_address(const char *name)
{
    return strlen(name) == PCI_ADDRESS_LENGTH && name[4] == ':' && name[7] == ':' &&
           name[10] == '.';
}

/** Copy an address the caller has already checked with looks_like_pci_address(). */
static void copy_pci_address(char destination[FT_TT_PCI_ADDRESS_MAX], const char *name)
{
    memcpy(destination, name, PCI_ADDRESS_LENGTH);
    destination[PCI_ADDRESS_LENGTH] = '\0';
}

/* Defined with the rung-5 code below; used by both halves. */
static bool wait_for_device(const char *bdf, unsigned timeout_milliseconds);
static bool driver_is_bound(const char *bdf);

/** Is the card's memory decode enabled? A reset clears the command register, and a card
 *  with `Mem-` answers every BAR access as if it were not there. */
static bool memory_decode_enabled(const char *bdf)
{
    uint16_t command = 0;

    if (read_config_word(bdf, PCI_COMMAND_OFFSET, &command) != FT_TT_OK)
        return false;
    return (command & PCI_COMMAND_MEMORY) != 0u;
}

int ft_tt_pcie_reenumerate_is_permitted(const char *card_bdf)
{
    char control_path[256];

    if (!card_bdf)
        return FT_TT_EINVAL;
    /* W_OK on the sysfs control node, which is root-only. access() rather than geteuid()
     * so a capability or an ACL that grants it without uid 0 is honoured, and so the
     * answer is about the file we will actually write. */
    if (snprintf(control_path, sizeof(control_path), SYSFS_PCI_DEVICES "/%s/remove", card_bdf) < 0)
        return FT_TT_EIO;
    if (access(control_path, W_OK) != 0)
        return FT_TT_EPERM;
    if (access("/sys/bus/pci/rescan", W_OK) != 0)
        return FT_TT_EPERM;
    return FT_TT_OK;
}

int ft_tt_pcie_reenumerate(const char *card_bdf, ft_tt_pcie_reenumeration_report *out)
{
    ft_tt_pcie_topology topology;
    char control_path[256];
    struct stat node;

    if (!card_bdf || !out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));

    /* The bridge is derived from the card, never named: a rescan is harmless, but the
     * path it is written to should still come from the device we are holding. */
    if (ft_tt_pcie_topology_resolve(card_bdf, &topology) != FT_TT_OK || topology.bridge[0] == 0)
        return sysfs_step_failed(&out->failure_stage, &out->failure_errno,
                                 "resolving the card's bridge");

    if (snprintf(control_path, sizeof(control_path), SYSFS_PCI_DEVICES "/%s/remove", card_bdf) < 0)
        return FT_TT_EIO;
    if (write_sysfs_text(control_path, "1") != FT_TT_OK)
        return sysfs_step_failed(&out->failure_stage, &out->failure_errno,
                                 "removing the card from the PCI bus");
    out->removed = true;
    sleep_milliseconds(2000u);

    if (snprintf(control_path, sizeof(control_path), SYSFS_PCI_DEVICES "/%s/rescan",
                 topology.bridge) < 0)
        return FT_TT_EIO;
    if (write_sysfs_text(control_path, "1") != FT_TT_OK)
        return sysfs_step_failed(&out->failure_stage, &out->failure_errno,
                                 "rescanning the bridge");
    out->rescanned = true;

    out->device_returned = wait_for_device(card_bdf, SBR_REAPPEAR_MILLISECONDS);
    if (!out->device_returned)
        return sysfs_step_failed(&out->failure_stage, &out->failure_errno,
                                 "the card did not reappear after the rescan");
    sleep_milliseconds(2000u);

    out->bar0_length_after = read_bar0_length(card_bdf);
    out->memory_decode_enabled = memory_decode_enabled(card_bdf);
    out->driver_bound = driver_is_bound(card_bdf);
    out->chardev_present = stat("/dev/tenstorrent/0", &node) == 0;
    return FT_TT_OK;
}

void ft_tt_pcie_reenumeration_print(const ft_tt_pcie_reenumeration_report *report)
{
    if (!report)
        return;
    printf("re-enumeration: removed=%s rescanned=%s returned=%s\n",
           report->removed ? "yes" : "no", report->rescanned ? "yes" : "no",
           report->device_returned ? "yes" : "no");
    printf("  BAR0 %llu bytes, memory decode %s, tt-kmd %s, /dev/tenstorrent/0 %s\n",
           (unsigned long long)report->bar0_length_after,
           report->memory_decode_enabled ? "ENABLED" : "DISABLED",
           report->driver_bound ? "bound" : "NOT BOUND",
           report->chardev_present ? "present" : "MISSING");
    if (report->failure_stage)
        printf("  FAILED at: %s (errno %d)\n", report->failure_stage, report->failure_errno);
}

/* ================================================================
 * Topology — the refusal
 * ================================================================ */

int ft_tt_pcie_topology_resolve(const char *card_bdf, ft_tt_pcie_topology *out)
{
    char device_link[256];
    char resolved[PATH_MAX];
    char *last_separator;
    char vendor_text[32];
    char path[256];
    DIR *bridge_directory;
    struct dirent *entry;
    unsigned long parsed;

    if (!card_bdf || !out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));
    if (!looks_like_pci_address(card_bdf))
        return FT_TT_EINVAL;
    copy_pci_address(out->card, card_bdf);

    /* The bridge is DERIVED from the card's own device path. It is never named from
     * outside this module, so a typo cannot aim this at a bridge with K80s under it. */
    if (snprintf(device_link, sizeof(device_link), SYSFS_PCI_DEVICES "/%s", card_bdf) < 0)
        return FT_TT_EIO;
    if (!realpath(device_link, resolved)) {
        out->refusal = "the card has no sysfs device path";
        return FT_TT_EIO;
    }
    last_separator = strrchr(resolved, '/');
    if (!last_separator) {
        out->refusal = "the card's device path has no parent";
        return FT_TT_EIO;
    }
    *last_separator = '\0';
    last_separator = strrchr(resolved, '/');
    if (!last_separator || !looks_like_pci_address(last_separator + 1)) {
        out->refusal = "the card's parent is not a PCI bridge (it is a host bridge or root)";
        return FT_TT_OK;
    }
    copy_pci_address(out->bridge, last_separator + 1);

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s/vendor", card_bdf) < 0)
        return FT_TT_EIO;
    if (read_sysfs_text(path, vendor_text, sizeof(vendor_text)) == FT_TT_OK) {
        parsed = strtoul(vendor_text, NULL, 0);
        out->card_vendor_id = (uint16_t)parsed;
    }
    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s/device", card_bdf) < 0)
        return FT_TT_EIO;
    if (read_sysfs_text(path, vendor_text, sizeof(vendor_text)) == FT_TT_OK) {
        parsed = strtoul(vendor_text, NULL, 0);
        out->card_device_id = (uint16_t)parsed;
    }

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s", out->bridge) < 0)
        return FT_TT_EIO;
    bridge_directory = opendir(path);
    if (!bridge_directory) {
        out->refusal = "the bridge has no sysfs directory";
        return FT_TT_EIO;
    }
    while ((entry = readdir(bridge_directory)) != NULL) {
        if (!looks_like_pci_address(entry->d_name))
            continue;
        out->child_count++;
        if (strcmp(entry->d_name, card_bdf) != 0 && out->unexpected_child[0] == '\0')
            copy_pci_address(out->unexpected_child, entry->d_name);
    }
    closedir(bridge_directory);

    if (out->child_count != 1u)
        out->refusal = "the bridge does not have exactly one child device";
    else if (out->unexpected_child[0] != '\0')
        out->refusal = "the bridge's only child is not the card";
    else if (out->card_vendor_id != FT_TT_PCI_VENDOR_ID)
        out->refusal = "the card's PCI vendor id is not Tenstorrent's";
    else
        out->safe = true;
    return FT_TT_OK;
}

void ft_tt_pcie_topology_print(const ft_tt_pcie_topology *topology)
{
    if (!topology)
        return;
    printf("PCIe topology: card %s [%04x:%04x] behind bridge %s, %u child device%s\n",
           topology->card, topology->card_vendor_id, topology->card_device_id,
           topology->bridge[0] ? topology->bridge : "(none)", topology->child_count,
           topology->child_count == 1u ? "" : "s");
    if (topology->unexpected_child[0] != '\0')
        printf("  also behind that bridge: %s\n", topology->unexpected_child);
    if (topology->safe)
        printf("  a secondary bus reset here touches the card and nothing else\n");
    else
        printf("  REFUSED: %s\n", topology->refusal ? topology->refusal : "unknown");
}

/* ================================================================
 * Rung 5 — the secondary bus reset
 * ================================================================ */

/** Wait for the device directory to reappear after a rescan. */
static bool wait_for_device(const char *bdf, unsigned timeout_milliseconds)
{
    char path[256];
    unsigned waited = 0;
    struct stat node;

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s", bdf) < 0)
        return false;
    while (waited < timeout_milliseconds) {
        if (stat(path, &node) == 0)
            return true;
        sleep_milliseconds(100u);
        waited += 100u;
    }
    return stat(path, &node) == 0;
}

/** Is a driver bound to the device now? */
static bool driver_is_bound(const char *bdf)
{
    char path[256];
    char target[256];
    ssize_t length;

    if (snprintf(path, sizeof(path), SYSFS_PCI_DEVICES "/%s/driver", bdf) < 0)
        return false;
    length = readlink(path, target, sizeof(target) - 1u);
    if (length <= 0)
        return false;
    target[length] = '\0';
    return strstr(target, "tenstorrent") != NULL;
}

/** Identity words (vendor, device) survive re-enumeration; BAR assignment need not. */
static bool identity_matches(const char *bdf, const ft_tt_pcie_topology *expected)
{
    ft_tt_pcie_topology now;

    if (ft_tt_pcie_topology_resolve(bdf, &now) != FT_TT_OK)
        return false;
    return now.card_vendor_id == expected->card_vendor_id &&
           now.card_device_id == expected->card_device_id;
}

static int sbr_fail(ft_tt_pcie_sbr_report *out, const char *stage, int error_number)
{
    out->failure_stage = stage;
    out->failure_errno = error_number;
    return FT_TT_EIO;
}

int ft_tt_pcie_secondary_bus_reset(const char *card_bdf, const char *save_directory,
                                   ft_tt_pcie_sbr_report *out)
{
    char control_path[256];
    struct stat node;
    uint16_t control;
    int rc;

    if (!card_bdf || !out)
        return FT_TT_EINVAL;
    memset(out, 0, sizeof(*out));

    rc = ft_tt_pcie_topology_resolve(card_bdf, &out->topology);
    if (rc != FT_TT_OK)
        return sbr_fail(out, "reading the PCIe topology", 0);
    if (!out->topology.safe) {
        out->failure_stage = "topology check refused the reset";
        return FT_TT_EPERM;
    }

    out->bar0_length_before = read_bar0_length(card_bdf);

    if (save_directory) {
        if (save_config_space(card_bdf, save_directory, "card") != FT_TT_OK)
            return sbr_fail(out, "saving the card's config space", errno);
        if (save_config_space(out->topology.bridge, save_directory, "bridge") != FT_TT_OK)
            return sbr_fail(out, "saving the bridge's config space", errno);
        out->config_saved = true;
    }

    if (read_config_word(out->topology.bridge, PCI_BRIDGE_CONTROL_OFFSET,
                         &out->bridge_control_before) != FT_TT_OK)
        return sbr_fail(out, "reading the bridge control register", errno);

    /* Remove the card first: the reset is about to invalidate everything tt-kmd knows
     * about it, and a driver still bound through that is a use-after-reset. */
    if (snprintf(control_path, sizeof(control_path), SYSFS_PCI_DEVICES "/%s/remove", card_bdf) < 0)
        return sbr_fail(out, "building the remove path", 0);
    if (write_sysfs_text(control_path, "1") != FT_TT_OK)
        return sbr_fail(out, "removing the card from the PCI bus", errno);
    out->device_removed = true;
    sleep_milliseconds(200u);

    control = (uint16_t)(out->bridge_control_before | PCI_BRIDGE_CONTROL_SECONDARY_BUS_RESET);
    if (write_config_word(out->topology.bridge, PCI_BRIDGE_CONTROL_OFFSET, control) != FT_TT_OK)
        return sbr_fail(out, "asserting secondary bus reset", errno);
    (void)read_config_word(out->topology.bridge, PCI_BRIDGE_CONTROL_OFFSET,
                           &out->bridge_control_asserted);
    sleep_milliseconds(SBR_ASSERT_MILLISECONDS);

    /* Restore the register exactly as it was — this is the one register we modified. */
    if (write_config_word(out->topology.bridge, PCI_BRIDGE_CONTROL_OFFSET,
                          out->bridge_control_before) != FT_TT_OK)
        return sbr_fail(out, "deasserting secondary bus reset", errno);
    (void)read_config_word(out->topology.bridge, PCI_BRIDGE_CONTROL_OFFSET,
                           &out->bridge_control_after);
    sleep_milliseconds(SBR_SETTLE_MILLISECONDS);

    if (snprintf(control_path, sizeof(control_path), SYSFS_PCI_DEVICES "/%s/rescan",
                 out->topology.bridge) < 0)
        return sbr_fail(out, "building the rescan path", 0);
    if (write_sysfs_text(control_path, "1") != FT_TT_OK)
        return sbr_fail(out, "rescanning the bridge", errno);
    out->rescanned = true;

    out->device_returned = wait_for_device(card_bdf, SBR_REAPPEAR_MILLISECONDS);
    if (!out->device_returned)
        return sbr_fail(out, "the card did not reappear after the rescan", 0);

    out->bar0_length_after = read_bar0_length(card_bdf);
    out->driver_bound = driver_is_bound(card_bdf);
    out->config_matches_saved = identity_matches(card_bdf, &out->topology);
    out->chardev_present = stat("/dev/tenstorrent/0", &node) == 0;
    return FT_TT_OK;
}

void ft_tt_pcie_sbr_print(const ft_tt_pcie_sbr_report *report)
{
    if (!report)
        return;

    ft_tt_pcie_topology_print(&report->topology);
    printf("  config space saved:   %s\n", report->config_saved ? "yes" : "no");
    printf("  card removed:         %s\n", report->device_removed ? "yes" : "no");
    printf("  bridge control:       0x%04x -> 0x%04x (asserted) -> 0x%04x\n",
           report->bridge_control_before, report->bridge_control_asserted,
           report->bridge_control_after);
    printf("  bridge rescanned:     %s\n", report->rescanned ? "yes" : "no");
    printf("  card returned:        %s\n", report->device_returned ? "yes" : "no");
    printf("  BAR0 length:          %llu -> %llu bytes\n",
           (unsigned long long)report->bar0_length_before,
           (unsigned long long)report->bar0_length_after);
    printf("  identity unchanged:   %s\n", report->config_matches_saved ? "yes" : "no");
    printf("  tt-kmd bound:         %s\n", report->driver_bound ? "yes" : "no");
    printf("  /dev/tenstorrent/0:   %s\n", report->chardev_present ? "present" : "MISSING");
    if (report->failure_stage)
        printf("  FAILED at: %s (errno %d)\n", report->failure_stage, report->failure_errno);
}
