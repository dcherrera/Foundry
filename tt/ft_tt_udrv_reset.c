/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_reset.c
 * @brief Implementation of the userspace ASIC reset, PCI config snapshot/restore and PCIe
 *   link retrain. Every hardware access in this file is a `pread`/`pwrite` on a sysfs
 *   `config` file, which is what makes the whole module testable against a fixture
 *   directory.
 *
 * @status See ft_tt_udrv_reset.h.
 * @issues None
 * @todo None
 */

/* `pread`/`pwrite`, `O_CLOEXEC`, `clock_gettime` and `nanosleep` are POSIX.1-2008, and
 * -std=c99 hides all four under glibc. clang on macOS exposes them anyway, which is exactly
 * how a Mac-side build passes and the Linux card host's gcc fails. */
#define _POSIX_C_SOURCE 200809L

#include "ft_tt_udrv_reset.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_udrv.h"

/* Config-space offsets. Named here rather than included from a kernel header so this module
 * carries no dependency on the layer it replaces. */
#define CONFIG_VENDOR_ID 0x00u
#define CONFIG_DEVICE_ID 0x02u
#define CONFIG_COMMAND 0x04u
#define CONFIG_CAPABILITY_POINTER 0x34u

/* The interface timer, as the kernel driver programs it. ORDER: target first, then control. */
#define INTERFACE_TIMER_CONTROL_OFFSET 0x930u
#define INTERFACE_TIMER_TARGET_OFFSET 0x934u
#define INTERFACE_TIMER_TARGET_VALUE 0x00000001u
#define INTERFACE_TIMER_FIRE_VALUE 0x00000011u /* EN | FORCE_PENDING */

/* Byte offsets inside the PCI Express capability. */
#define PCIE_FLAGS 0x02u
#define PCIE_DEVCTL 0x08u
#define PCIE_LNKCTL 0x10u
#define PCIE_LNKSTA 0x12u
#define PCIE_SLTCTL 0x18u
#define PCIE_RTCTL 0x1Cu
#define PCIE_DEVCTL2 0x28u
#define PCIE_LNKCTL2 0x30u
#define PCIE_SLTCTL2 0x38u

/** The saved words, in `pci_save_pcie_state()` order. Index 4..6 exist only at capability
 *  version 2 or above, which is why the table carries the minimum version. */
static const struct {
    unsigned offset;
    unsigned minimum_capability_version;
    uint16_t verify_mask;
} pcie_saved_register[FT_TT_UDRV_PCIE_SAVED_WORDS] = {
    {PCIE_DEVCTL, 1u, 0xFFFFu},
    /* Retrain Link is write-1, self-clearing: it never reads back as written. */
    {PCIE_LNKCTL, 1u, (uint16_t)~FT_TT_UDRV_LNKCTL_RETRAIN},
    {PCIE_SLTCTL, 1u, 0xFFFFu},
    {PCIE_RTCTL, 1u, 0xFFFFu},
    {PCIE_DEVCTL2, 2u, 0xFFFFu},
    {PCIE_LNKCTL2, 2u, 0xFFFFu},
    {PCIE_SLTCTL2, 2u, 0xFFFFu}};

/** The capability list lives in the legacy 256-byte window and starts after the header. */
#define CAPABILITY_LIST_FIRST 0x40u
#define CAPABILITY_LIST_LAST 0xFCu
#define CAPABILITY_WALK_LIMIT 64u

/* ------------------------------------------------------------------------ *
 * Raw access
 * ------------------------------------------------------------------------ */

/** Build "<device_dir>/config" or refuse if it will not fit. */
static int config_path(const char *device_dir, char *out, size_t out_bytes)
{
    int written;

    if (!device_dir || !out)
        return FT_TT_UDRV_EINVAL;
    written = snprintf(out, out_bytes, "%s/config", device_dir);
    if (written < 0 || (size_t)written >= out_bytes)
        return FT_TT_UDRV_EINVAL;
    return FT_TT_UDRV_OK;
}

/** One pread of `width` bytes at `offset`, assembled little-endian. */
static int config_read(const char *device_dir, unsigned offset, unsigned width,
                       uint32_t *out_value)
{
    char path[FT_TT_UDRV_RESET_PATH_MAX];
    unsigned char bytes[4] = {0, 0, 0, 0};
    uint32_t value = 0u;
    unsigned byte;
    ssize_t got;
    int fd;
    int rc;

    if (!out_value || (width != 2u && width != 4u) || (offset % width) != 0u)
        return FT_TT_UDRV_EINVAL;
    rc = config_path(device_dir, path, sizeof(path));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return FT_TT_UDRV_EIO;
    got = pread(fd, bytes, (size_t)width, (off_t)offset);
    close(fd);
    if (got != (ssize_t)width)
        return FT_TT_UDRV_EIO;

    for (byte = 0u; byte < width; byte++)
        value |= (uint32_t)bytes[byte] << (8u * byte);
    *out_value = value;
    return FT_TT_UDRV_OK;
}

/** One pwrite of `width` bytes at `offset`, little-endian. */
static int config_write(const char *device_dir, unsigned offset, unsigned width, uint32_t value)
{
    char path[FT_TT_UDRV_RESET_PATH_MAX];
    unsigned char bytes[4];
    unsigned byte;
    ssize_t put;
    int fd;
    int rc;

    if ((width != 2u && width != 4u) || (offset % width) != 0u)
        return FT_TT_UDRV_EINVAL;
    rc = config_path(device_dir, path, sizeof(path));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    for (byte = 0u; byte < width; byte++)
        bytes[byte] = (unsigned char)((value >> (8u * byte)) & 0xFFu);

    fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0)
        return FT_TT_UDRV_EIO;
    put = pwrite(fd, bytes, (size_t)width, (off_t)offset);
    close(fd);
    return put == (ssize_t)width ? FT_TT_UDRV_OK : FT_TT_UDRV_EIO;
}

int ft_tt_udrv_config_read_dword(const char *device_dir, unsigned offset, uint32_t *out_value)
{
    return config_read(device_dir, offset, 4u, out_value);
}

int ft_tt_udrv_config_read_word(const char *device_dir, unsigned offset, uint16_t *out_value)
{
    uint32_t wide = 0u;
    int rc;

    if (!out_value)
        return FT_TT_UDRV_EINVAL;
    rc = config_read(device_dir, offset, 2u, &wide);
    if (rc == FT_TT_UDRV_OK)
        *out_value = (uint16_t)wide;
    return rc;
}

int ft_tt_udrv_config_write_dword(const char *device_dir, unsigned offset, uint32_t value)
{
    return config_write(device_dir, offset, 4u, value);
}

int ft_tt_udrv_config_write_word(const char *device_dir, unsigned offset, uint16_t value)
{
    return config_write(device_dir, offset, 2u, value);
}

int ft_tt_udrv_config_find_capability(const char *device_dir, unsigned capability_id,
                                      unsigned *out_offset)
{
    unsigned char seen[CAPABILITY_LIST_LAST + 1u];
    uint32_t pointer_dword = 0u;
    unsigned offset;
    unsigned step;
    int rc;

    if (!out_offset)
        return FT_TT_UDRV_EINVAL;
    memset(seen, 0, sizeof(seen));

    /* The capability pointer is a byte at 0x34; read the dword it lives in and mask, so
     * every access in this module stays 2- or 4-byte aligned. */
    rc = ft_tt_udrv_config_read_dword(device_dir, CONFIG_CAPABILITY_POINTER, &pointer_dword);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    offset = pointer_dword & 0xFCu;

    for (step = 0u; step < CAPABILITY_WALK_LIMIT; step++) {
        uint32_t entry = 0u;

        if (offset == 0u)
            return FT_TT_UDRV_ENODEV; /* a well-formed list that simply lacks it */
        if (offset < CAPABILITY_LIST_FIRST || offset > CAPABILITY_LIST_LAST ||
            (offset % 4u) != 0u || seen[offset])
            return FT_TT_UDRV_EIO; /* malformed or looping — refuse, never follow */
        seen[offset] = 1u;

        rc = ft_tt_udrv_config_read_dword(device_dir, offset, &entry);
        if (rc != FT_TT_UDRV_OK)
            return rc;
        if ((entry & 0xFFu) == capability_id) {
            *out_offset = offset;
            return FT_TT_UDRV_OK;
        }
        offset = (entry >> 8u) & 0xFCu;
    }
    return FT_TT_UDRV_EIO;
}

/* ------------------------------------------------------------------------ *
 * MRRS
 * ------------------------------------------------------------------------ */

unsigned ft_tt_udrv_max_read_request_bytes(unsigned code)
{
    /* Codes 6 and 7 are reserved. Answering 8192/16384 for them would invent a
     * size the spec does not define, and this number is printed as evidence. */
    if (code > FT_TT_UDRV_MRRS_CODE_MAX)
        return 0u;
    return 128u << code;
}

/** DEVCTL bits 14:12 out of a whole DEVCTL word. */
static unsigned mrrs_code_of(uint16_t devctl)
{
    return (unsigned)((devctl & FT_TT_UDRV_MRRS_MASK) >> FT_TT_UDRV_MRRS_SHIFT);
}

/** The code that expresses `bytes`, or FT_TT_UDRV_MRRS_CODE_MAX + 1 if none does. */
static unsigned mrrs_code_for_bytes(unsigned bytes)
{
    unsigned code;

    for (code = 0u; code <= FT_TT_UDRV_MRRS_CODE_MAX; code++) {
        if (ft_tt_udrv_max_read_request_bytes(code) == bytes)
            return code;
    }
    return FT_TT_UDRV_MRRS_CODE_MAX + 1u;
}

int ft_tt_udrv_config_max_read_request_read(const char *device_dir, unsigned *out_code,
                                            unsigned *out_bytes)
{
    unsigned capability_offset = 0u;
    uint16_t devctl = 0u;
    int rc;

    if (!out_code)
        return FT_TT_UDRV_EINVAL;
    rc = ft_tt_udrv_config_find_capability(device_dir, FT_TT_UDRV_CAP_ID_EXPRESS,
                                           &capability_offset);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    rc = ft_tt_udrv_config_read_word(device_dir, capability_offset + PCIE_DEVCTL, &devctl);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    *out_code = mrrs_code_of(devctl);
    if (out_bytes)
        *out_bytes = ft_tt_udrv_max_read_request_bytes(*out_code);
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_config_max_read_request_set(const char *device_dir, unsigned bytes,
                                           ft_tt_udrv_max_read_request_report *out)
{
    unsigned wanted = mrrs_code_for_bytes(bytes);
    uint16_t devctl = 0u;
    uint16_t intended;
    int rc;

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));
    if (wanted > FT_TT_UDRV_MRRS_CODE_MAX) {
        out->failure_stage = "size is not a legal Max_Read_Request_Size";
        return FT_TT_UDRV_EINVAL;
    }

    rc = ft_tt_udrv_config_find_capability(device_dir, FT_TT_UDRV_CAP_ID_EXPRESS,
                                           &out->capability_offset);
    if (rc != FT_TT_UDRV_OK) {
        /* Named precisely, because the common cause is NOT a card without the
         * capability: an unprivileged reader sees only the 64-byte standard header,
         * so the pointer at 0x34 reads and the capability it points at does not. */
        out->failure_stage = "could not walk to the PCI Express capability (an unprivileged "
                             "reader sees only the 64-byte config header)";
        out->failure_errno = errno;
        return rc;
    }
    rc = ft_tt_udrv_config_read_word(device_dir, out->capability_offset + PCIE_DEVCTL, &devctl);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "read DEVCTL";
        out->failure_errno = errno;
        return rc;
    }

    out->devctl_before = devctl;
    out->code_before = mrrs_code_of(devctl);
    out->bytes_before = ft_tt_udrv_max_read_request_bytes(out->code_before);
    /* The `after` fields start as the state that was actually READ, so every exit
     * path below reports something observed rather than a zero that would print
     * as a plausible "128 bytes". */
    out->devctl_after = devctl;
    out->code_after = out->code_before;
    out->bytes_after = out->bytes_before;

    if (out->code_before == wanted) {
        out->already_correct = true;
        return FT_TT_UDRV_OK;
    }

    /* Read-modify-write: DEVCTL also carries the four error-reporting enables,
     * Relaxed Ordering, No Snoop and Max_Payload, and a bare store of the MRRS
     * field alone would clear whatever the previous owner set. */
    intended = (uint16_t)((devctl & (uint16_t) ~FT_TT_UDRV_MRRS_MASK) |
                          (uint16_t)(wanted << FT_TT_UDRV_MRRS_SHIFT));
    rc = ft_tt_udrv_config_write_word(device_dir, out->capability_offset + PCIE_DEVCTL, intended);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "write DEVCTL";
        out->failure_errno = errno;
        return rc;
    }

    /* A real config read of the register just written. None of DEVCTL's bits are
     * RW1C, so it must read back exactly; anything else is a refusal. */
    rc = ft_tt_udrv_config_read_word(device_dir, out->capability_offset + PCIE_DEVCTL, &devctl);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "read DEVCTL back";
        out->failure_errno = errno;
        return rc;
    }
    out->devctl_after = devctl;
    out->code_after = mrrs_code_of(devctl);
    out->bytes_after = ft_tt_udrv_max_read_request_bytes(out->code_after);
    if (out->code_after != wanted) {
        out->failure_stage = "DEVCTL did not hold the requested Max_Read_Request_Size";
        return FT_TT_UDRV_EIO;
    }
    out->written = true;
    return FT_TT_UDRV_OK;
}

/* PCI_COMMAND bit 1. Named here rather than pulled from a kernel header because
 * this file speaks to `config` through pread/pwrite and includes nothing of the
 * kernel's. */
#define CONFIG_COMMAND_MEMORY_SPACE 0x0002u

int ft_tt_udrv_config_memory_decode_enable(const char *device_dir,
                                           ft_tt_udrv_memory_decode_report *out)
{
    uint16_t command = 0u;
    int rc;

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));

    rc = ft_tt_udrv_config_read_word(device_dir, CONFIG_COMMAND, &command);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "read PCI_COMMAND";
        out->failure_errno = errno;
        return rc;
    }
    out->command_before = command;
    /* `after` starts as the word that was READ, so a failure below reports an
     * observed value rather than a zero that would print as a plausible state. */
    out->command_after = command;

    if ((command & CONFIG_COMMAND_MEMORY_SPACE) != 0u) {
        out->already_enabled = true;
        return FT_TT_UDRV_OK;
    }

    rc = ft_tt_udrv_config_write_word(device_dir, CONFIG_COMMAND,
                                      (uint16_t)(command | CONFIG_COMMAND_MEMORY_SPACE));
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "write PCI_COMMAND";
        out->failure_errno = errno;
        return rc;
    }

    /* A real config read of the register just written. PCI_COMMAND's low bits are
     * plain RW, so it must read back with the bit set; anything else is a refusal.
     * Bit 1 alone is tested rather than the whole word, because the RW1C status
     * half above it can change under us between the write and the read. */
    rc = ft_tt_udrv_config_read_word(device_dir, CONFIG_COMMAND, &command);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "read PCI_COMMAND back";
        out->failure_errno = errno;
        return rc;
    }
    out->command_after = command;
    if ((command & CONFIG_COMMAND_MEMORY_SPACE) == 0u) {
        out->failure_stage = "PCI_COMMAND did not hold Memory Space Enable";
        return FT_TT_UDRV_EIO;
    }
    out->written = true;
    return FT_TT_UDRV_OK;
}

void ft_tt_udrv_max_read_request_print(const ft_tt_udrv_max_read_request_report *report)
{
    if (!report)
        return;
    if (report->failure_stage) {
        printf("MRRS: %s (errno %d); it still reads %u bytes (code %u)\n", report->failure_stage,
               report->failure_errno, report->bytes_after, report->code_after);
        return;
    }
    printf("MRRS: %u bytes (code %u) -> %u bytes (code %u)  [%s, DEVCTL 0x%04x -> 0x%04x]\n",
           report->bytes_before, report->code_before, report->bytes_after, report->code_after,
           report->written ? "WRITTEN by us" : "already correct, nothing written",
           report->devctl_before, report->devctl_after);
}

/* ------------------------------------------------------------------------ *
 * Snapshot and restore
 * ------------------------------------------------------------------------ */

int ft_tt_udrv_config_snapshot_take(const char *device_dir, ft_tt_udrv_config_snapshot *out)
{
    uint16_t flags = 0u;
    unsigned index;
    int rc;

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));

    for (index = 0u; index < FT_TT_UDRV_CONFIG_HEADER_DWORDS; index++) {
        rc = ft_tt_udrv_config_read_dword(device_dir, index * 4u, &out->header[index]);
        if (rc != FT_TT_UDRV_OK)
            return rc;
    }
    out->vendor_id = (uint16_t)(out->header[0] & 0xFFFFu);
    out->device_id = (uint16_t)((out->header[0] >> 16u) & 0xFFFFu);

    rc = ft_tt_udrv_config_find_capability(device_dir, FT_TT_UDRV_CAP_ID_EXPRESS,
                                           &out->pcie_capability_offset);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = ft_tt_udrv_config_read_word(device_dir, out->pcie_capability_offset + PCIE_FLAGS,
                                     &flags);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    out->pcie_capability_version = (unsigned)(flags & 0x000Fu);

    for (index = 0u; index < FT_TT_UDRV_PCIE_SAVED_WORDS; index++) {
        if (out->pcie_capability_version < pcie_saved_register[index].minimum_capability_version)
            continue;
        rc = ft_tt_udrv_config_read_word(
            device_dir, out->pcie_capability_offset + pcie_saved_register[index].offset,
            &out->pcie_word[index]);
        if (rc != FT_TT_UDRV_OK)
            return rc;
        out->pcie_word_present[index] = true;
    }
    out->valid = true;
    return FT_TT_UDRV_OK;
}

uint32_t ft_tt_udrv_config_header_verify_mask(unsigned index)
{
    return index == 1u ? 0x0000FFFFu : 0xFFFFFFFFu;
}

/** Write one register if it differs, verify it, and journal the step. */
static int restore_one(const char *device_dir, ft_tt_udrv_config_restore_report *report,
                       unsigned offset, unsigned width, uint32_t value, uint32_t verify_mask,
                       const char *stage)
{
    ft_tt_udrv_config_step *step;
    uint32_t current = 0u;
    int rc;

    if (report->step_count >= FT_TT_UDRV_RESTORE_STEPS_MAX) {
        report->failure_stage = "restore journal overflow";
        return FT_TT_UDRV_ERROR;
    }
    step = &report->step[report->step_count++];
    step->offset = offset;
    step->width = width;
    step->value = value;

    rc = config_read(device_dir, offset, width, &current);
    if (rc != FT_TT_UDRV_OK) {
        report->failure_stage = stage;
        report->failure_errno = errno;
        return rc;
    }
    if (((current ^ value) & verify_mask) == 0u) {
        step->skipped = true;
        return FT_TT_UDRV_OK;
    }

    rc = config_write(device_dir, offset, width, value);
    if (rc != FT_TT_UDRV_OK) {
        report->failure_stage = stage;
        report->failure_errno = errno;
        return rc;
    }
    report->written++;

    rc = config_read(device_dir, offset, width, &current);
    if (rc != FT_TT_UDRV_OK) {
        report->failure_stage = stage;
        report->failure_errno = errno;
        return rc;
    }
    if (((current ^ value) & verify_mask) != 0u) {
        if (report->mismatches == 0u)
            report->first_mismatch_offset = offset;
        report->mismatches++;
    }
    return FT_TT_UDRV_OK;
}

/** Header dwords `first..last` inclusive, ascending — one of the kernel's three ranges. */
static int restore_header_range(const char *device_dir, const ft_tt_udrv_config_snapshot *snapshot,
                                ft_tt_udrv_config_restore_report *report, unsigned first,
                                unsigned last)
{
    unsigned index;

    for (index = first; index <= last; index++) {
        int rc = restore_one(device_dir, report, index * 4u, 4u, snapshot->header[index],
                             ft_tt_udrv_config_header_verify_mask(index),
                             "restoring a header dword");

        if (rc != FT_TT_UDRV_OK)
            return rc;
    }
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_config_snapshot_restore(const char *device_dir,
                                       const ft_tt_udrv_config_snapshot *snapshot,
                                       ft_tt_udrv_config_restore_report *out)
{
    unsigned index;
    int rc;

    if (!snapshot || !out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));
    if (!snapshot->valid) {
        out->failure_stage = "the snapshot is not valid";
        return FT_TT_UDRV_EINVAL;
    }

    /* THE ORDER IS pci_restore_state()'s. Capability words first; then header dwords 10-15,
     * then the BARs (4-9), then 0-3 — so COMMAND lands after the BARs it enables. */
    for (index = 0u; index < FT_TT_UDRV_PCIE_SAVED_WORDS; index++) {
        if (!snapshot->pcie_word_present[index])
            continue;
        rc = restore_one(device_dir, out,
                         snapshot->pcie_capability_offset + pcie_saved_register[index].offset, 2u,
                         snapshot->pcie_word[index], pcie_saved_register[index].verify_mask,
                         "restoring a PCI Express capability word");
        if (rc != FT_TT_UDRV_OK)
            return rc;
    }

    rc = restore_header_range(device_dir, snapshot, out, 10u, 15u);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    rc = restore_header_range(device_dir, snapshot, out, 4u, 9u);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    rc = restore_header_range(device_dir, snapshot, out, 0u, 3u);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (out->mismatches != 0u) {
        out->failure_stage = "a restored register did not read back";
        return FT_TT_UDRV_EIO;
    }
    return FT_TT_UDRV_OK;
}

void ft_tt_udrv_config_restore_print(const ft_tt_udrv_config_restore_report *report)
{
    unsigned index;

    if (!report)
        return;
    printf("config restore: %u step%s, %u written, %u skipped, %u mismatch%s\n",
           report->step_count, report->step_count == 1u ? "" : "s", report->written,
           report->step_count - report->written, report->mismatches,
           report->mismatches == 1u ? "" : "es");
    for (index = 0u; index < report->step_count; index++) {
        const ft_tt_udrv_config_step *step = &report->step[index];

        printf("  %2u. 0x%03x %u-byte <- 0x%08x%s\n", index, step->offset, step->width,
               step->value, step->skipped ? "  (already held it)" : "");
    }
    if (report->failure_stage)
        printf("  FAILED at %s (errno %d), first mismatch at 0x%03x\n", report->failure_stage,
               report->failure_errno, report->first_mismatch_offset);
}

/* ------------------------------------------------------------------------ *
 * The resets
 * ------------------------------------------------------------------------ */

/** Journal one reset write, then issue it. The journal entry is appended BEFORE the write so
 *  a failing write still names itself in the record. */
static int reset_write(const char *device_dir, ft_tt_udrv_reset_report *out, unsigned offset,
                       unsigned width, uint32_t value)
{
    if (out->step_count >= FT_TT_UDRV_RESET_STEPS_MAX)
        return FT_TT_UDRV_ERROR;
    out->step[out->step_count].offset = offset;
    out->step[out->step_count].width = width;
    out->step[out->step_count].value = value;
    out->step_count++;
    return config_write(device_dir, offset, width, value);
}

/** The two interface-timer writes, shared by both reset flavours. ORDER IS LOAD-BEARING:
 *  0x934 arms the timer target and 0x930 fires it, so firing first resets against whatever
 *  target the previous owner of the card left behind. */
static int fire_interface_timer(const char *device_dir, ft_tt_udrv_reset_report *out)
{
    int rc = reset_write(device_dir, out, INTERFACE_TIMER_TARGET_OFFSET, 4u,
                         INTERFACE_TIMER_TARGET_VALUE);

    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "arming the interface timer (config 0x934)";
        out->failure_errno = errno;
        return rc;
    }
    out->timer_target_written = true;

    rc = reset_write(device_dir, out, INTERFACE_TIMER_CONTROL_OFFSET, 4u,
                     INTERFACE_TIMER_FIRE_VALUE);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "firing the interface timer (config 0x930)";
        out->failure_errno = errno;
        return rc;
    }
    out->timer_control_written = true;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_reset_asic(const char *device_dir, ft_tt_udrv_reset_report *out)
{
    int rc;

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));

    rc = ft_tt_udrv_config_read_word(device_dir, CONFIG_COMMAND, &out->command_before);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "reading PCI_COMMAND";
        out->failure_errno = errno;
        return rc;
    }
    out->command_with_marker = (uint16_t)(out->command_before | FT_TT_UDRV_COMMAND_PARITY);

    /* The marker MUST precede the timer: it is the only completion signal, and the card
     * clears it on its way out of reset. Setting it afterwards would set it on a card that
     * has already cleared it, and the wait would then never finish. */
    rc = reset_write(device_dir, out, CONFIG_COMMAND, 2u, out->command_with_marker);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "setting the reset marker (PCI_COMMAND.PARITY)";
        out->failure_errno = errno;
        return rc;
    }
    out->marker_set = true;

    return fire_interface_timer(device_dir, out);
}

int ft_tt_udrv_reset_config_write(const char *device_dir, ft_tt_udrv_reset_report *out)
{
    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));
    (void)ft_tt_udrv_config_read_word(device_dir, CONFIG_COMMAND, &out->command_before);
    out->command_with_marker = out->command_before; /* no marker in this flavour */
    return fire_interface_timer(device_dir, out);
}

/** Milliseconds since an arbitrary epoch, monotonic. */
static unsigned long monotonic_ms(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0ul;
    return (unsigned long)now.tv_sec * 1000ul + (unsigned long)(now.tv_nsec / 1000000l);
}

/** Sleep without pulling in nanosleep's EINTR handling at every call site. */
static void sleep_ms(unsigned milliseconds)
{
    struct timespec request;

    request.tv_sec = (time_t)(milliseconds / 1000u);
    request.tv_nsec = (long)(milliseconds % 1000u) * 1000000l;
    while (nanosleep(&request, &request) != 0 && errno == EINTR)
        continue;
}

int ft_tt_udrv_reset_wait_ready(const char *device_dir, uint16_t expect_vendor,
                                unsigned timeout_ms, ft_tt_udrv_reset_wait_report *out)
{
    const unsigned long start = monotonic_ms();

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));

    for (;;) {
        uint16_t vendor = 0xFFFFu;
        uint16_t command = 0xFFFFu;

        out->polls++;
        if (ft_tt_udrv_config_read_word(device_dir, CONFIG_VENDOR_ID, &vendor) == FT_TT_UDRV_OK)
            out->vendor_id_last = vendor;
        if (ft_tt_udrv_config_read_word(device_dir, CONFIG_COMMAND, &command) == FT_TT_UDRV_OK)
            out->command_last = command;

        /* BOTH conditions, and neither alone. A 0xFFFF vendor id means config reads are not
         * being answered; a marker still set means the ASIC has not come out of reset, and
         * restoring config at that instant is the bug that made the in-place path look
         * impossible (ft_tt_udrv_reset.h @status). */
        out->vendor_id_returned = (vendor == expect_vendor);
        out->marker_cleared = (command != 0xFFFFu) &&
                              ((command & FT_TT_UDRV_COMMAND_PARITY) == 0u);
        out->elapsed_ms = (unsigned)(monotonic_ms() - start);
        if (out->vendor_id_returned && out->marker_cleared)
            return FT_TT_UDRV_OK;
        if (out->elapsed_ms >= timeout_ms) {
            out->timed_out = true;
            return FT_TT_UDRV_EAGAIN;
        }
        sleep_ms(50u);
    }
}

void ft_tt_udrv_reset_print(const ft_tt_udrv_reset_report *reset,
                            const ft_tt_udrv_reset_wait_report *wait)
{
    if (reset) {
        unsigned index;

        printf("asic reset: PCI_COMMAND 0x%04x -> 0x%04x, marker %s, 0x934 %s, 0x930 %s\n",
               reset->command_before, reset->command_with_marker,
               reset->marker_set ? "set" : "NOT SET",
               reset->timer_target_written ? "armed" : "NOT ARMED",
               reset->timer_control_written ? "fired" : "NOT FIRED");
        for (index = 0u; index < reset->step_count; index++)
            printf("  %u. config 0x%03x %u-byte <- 0x%08x\n", index, reset->step[index].offset,
                   reset->step[index].width, reset->step[index].value);
        if (reset->failure_stage)
            printf("  FAILED at %s (errno %d)\n", reset->failure_stage, reset->failure_errno);
    }
    if (wait)
        printf("wait: %u ms, %u polls, vendor 0x%04x %s, command 0x%04x marker %s%s\n",
               wait->elapsed_ms, wait->polls, wait->vendor_id_last,
               wait->vendor_id_returned ? "answered" : "NOT ANSWERING", wait->command_last,
               wait->marker_cleared ? "cleared" : "STILL SET",
               wait->timed_out ? ", TIMED OUT" : "");
}

/* ------------------------------------------------------------------------ *
 * The link
 * ------------------------------------------------------------------------ */

/** Read one device's LNKSTA, having found its PCI Express capability. */
static int link_status_read_at(const char *device_dir, unsigned capability_offset,
                               ft_tt_udrv_link_status *out)
{
    uint16_t lnksta = 0u;
    int rc = ft_tt_udrv_config_read_word(device_dir, capability_offset + PCIE_LNKSTA, &lnksta);

    if (rc != FT_TT_UDRV_OK)
        return rc;
    out->speed_code = (unsigned)(lnksta & 0x000Fu);
    out->width = (unsigned)((lnksta >> 4u) & 0x003Fu);
    out->training = (lnksta & FT_TT_UDRV_LNKSTA_TRAINING) != 0u;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_link_status_read(const char *device_dir, ft_tt_udrv_link_status *out)
{
    unsigned capability_offset = 0u;
    int rc;

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));
    rc = ft_tt_udrv_config_find_capability(device_dir, FT_TT_UDRV_CAP_ID_EXPRESS,
                                           &capability_offset);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    return link_status_read_at(device_dir, capability_offset, out);
}

int ft_tt_udrv_link_retrain(const char *bridge_dir, unsigned timeout_ms,
                            ft_tt_udrv_link_retrain_report *out)
{
    const unsigned long start = monotonic_ms();
    unsigned capability_offset = 0u;
    uint16_t lnkctl = 0u;
    int rc;

    if (!out)
        return FT_TT_UDRV_EINVAL;
    memset(out, 0, sizeof(*out));

    rc = ft_tt_udrv_config_find_capability(bridge_dir, FT_TT_UDRV_CAP_ID_EXPRESS,
                                           &capability_offset);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "finding the bridge's PCI Express capability";
        out->failure_errno = errno;
        return rc;
    }
    rc = link_status_read_at(bridge_dir, capability_offset, &out->before);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "reading LNKSTA before the retrain";
        out->failure_errno = errno;
        return rc;
    }
    rc = ft_tt_udrv_config_read_word(bridge_dir, capability_offset + PCIE_LNKCTL, &lnkctl);
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "reading LNKCTL";
        out->failure_errno = errno;
        return rc;
    }
    /* Exactly what `setpci -s <bridge> CAP_EXP+10.w=0020:0020` writes. */
    rc = ft_tt_udrv_config_write_word(bridge_dir, capability_offset + PCIE_LNKCTL,
                                      (uint16_t)(lnkctl | FT_TT_UDRV_LNKCTL_RETRAIN));
    if (rc != FT_TT_UDRV_OK) {
        out->failure_stage = "setting LNKCTL.Retrain Link";
        out->failure_errno = errno;
        return rc;
    }
    out->retrain_requested = true;

    for (;;) {
        rc = link_status_read_at(bridge_dir, capability_offset, &out->after);
        if (rc != FT_TT_UDRV_OK) {
            out->failure_stage = "reading LNKSTA during the retrain";
            out->failure_errno = errno;
            return rc;
        }
        out->elapsed_ms = (unsigned)(monotonic_ms() - start);
        if (!out->after.training) {
            out->training_completed = true;
            return FT_TT_UDRV_OK;
        }
        if (out->elapsed_ms >= timeout_ms)
            return FT_TT_UDRV_EAGAIN;
        sleep_ms(10u);
    }
}

const char *ft_tt_udrv_link_speed_string(unsigned speed_code)
{
    switch (speed_code) {
    case 1u:
        return "2.5 GT/s";
    case 2u:
        return "5 GT/s";
    case 3u:
        return "8 GT/s";
    case 4u:
        return "16 GT/s";
    case 5u:
        return "32 GT/s";
    case 6u:
        return "64 GT/s";
    default:
        return "unknown";
    }
}

void ft_tt_udrv_link_retrain_print(const ft_tt_udrv_link_retrain_report *report)
{
    if (!report)
        return;
    printf("link retrain: %s x%u -> %s x%u (%s, %u ms)\n",
           ft_tt_udrv_link_speed_string(report->before.speed_code), report->before.width,
           ft_tt_udrv_link_speed_string(report->after.speed_code), report->after.width,
           report->training_completed ? "training complete" : "STILL TRAINING",
           report->elapsed_ms);
    if (report->failure_stage)
        printf("  FAILED at %s (errno %d)\n", report->failure_stage, report->failure_errno);
}
