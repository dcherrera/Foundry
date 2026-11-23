/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_pci_sysfs.c
 * @brief Implementation of sysfs-based PCI discovery for the userspace driver.
 *
 * @status Discovery and parsing are fixture-gated; ft_tt_udrv_pci_map_bar()
 *   has mapped BAR0 of the live p150a (2026-08-16).
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_udrv_pci_sysfs.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>
#include <unistd.h>

#include "ft_tt_udrv.h"

/* Linux `include/linux/ioport.h` resource flags, the ones `resource` can carry
 * for a BAR. Restated rather than included so this compiles on the Mac, where
 * the code is written. */
#define IORESOURCE_IO 0x00000100u
#define IORESOURCE_MEM 0x00000200u
#define IORESOURCE_PREFETCH 0x00002000u
#define IORESOURCE_MEM_64 0x00100000u

/** Longest single line the `resource` file can produce, with margin. */
#define RESOURCE_LINE_MAX 128u

/** Read a whole small sysfs attribute into `buffer`. */
static int read_attribute(const char *device_dir, const char *name, char *buffer,
                          size_t buffer_bytes)
{
    char path[FT_TT_UDRV_PCI_PATH_MAX];
    FILE *file;
    size_t length;
    int written;

    written = snprintf(path, sizeof(path), "%s/%s", device_dir, name);
    if (written < 0 || (size_t)written >= sizeof(path))
        return FT_TT_UDRV_EINVAL;

    file = fopen(path, "r");
    if (!file)
        return FT_TT_UDRV_ENODEV;

    if (fgets(buffer, (int)buffer_bytes, file) == NULL) {
        (void)fclose(file);
        return FT_TT_UDRV_ENODEV;
    }

    (void)fclose(file);

    length = strlen(buffer);
    while (length > 0u && (buffer[length - 1u] == '\n' || buffer[length - 1u] == '\r'))
        buffer[--length] = '\0';

    return (length > 0u) ? FT_TT_UDRV_OK : FT_TT_UDRV_ENODEV;
}

/**
 * Read an attribute holding one hexadecimal number ("0x1e52").
 *
 * strtoull with a checked end pointer, not sscanf: a file whose contents are
 * "0x" or "banana" must be a refusal, and sscanf("%x") reports success on the
 * first of those.
 */
static int read_hex_attribute(const char *device_dir, const char *name, uint64_t *out_value)
{
    char buffer[64];
    char *end = NULL;
    unsigned long long value;
    int rc;

    rc = read_attribute(device_dir, name, buffer, sizeof(buffer));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    errno = 0;
    value = strtoull(buffer, &end, 0);
    if (errno != 0 || end == buffer || (end && *end != '\0'))
        return FT_TT_UDRV_ENODEV;

    *out_value = (uint64_t)value;
    return FT_TT_UDRV_OK;
}

/** Decode the flag word into the booleans callers actually ask about. */
static void classify_bar(ft_tt_udrv_pci_bar *bar)
{
    /* The kernel's own pci_resource_len(): zero when `end` is zero. A BAR the
     * device does not implement prints "0x0 0x0 0x0", and a one-byte BAR would
     * have end == start != 0, so this rule and "end > start" are NOT the same
     * and only this one matches the kernel. */
    bar->length = (bar->end == 0u) ? 0u : (bar->end - bar->start + 1u);
    bar->present = (bar->length > 0u);
    bar->memory = (bar->flags & IORESOURCE_MEM) != 0u;
    bar->io = (bar->flags & IORESOURCE_IO) != 0u;
    bar->prefetchable = (bar->flags & IORESOURCE_PREFETCH) != 0u;
    bar->is_64bit = (bar->flags & IORESOURCE_MEM_64) != 0u;
}

/** Parse the first FT_TT_UDRV_PCI_BAR_COUNT lines of `resource`. */
static int read_resource_table(const char *device_dir, ft_tt_udrv_pci_device *out_device)
{
    char path[FT_TT_UDRV_PCI_PATH_MAX];
    char line[RESOURCE_LINE_MAX];
    FILE *file;
    unsigned index;
    int written;

    written = snprintf(path, sizeof(path), "%s/resource", device_dir);
    if (written < 0 || (size_t)written >= sizeof(path))
        return FT_TT_UDRV_EINVAL;

    file = fopen(path, "r");
    if (!file)
        return FT_TT_UDRV_ENODEV;

    for (index = 0u; index < FT_TT_UDRV_PCI_BAR_COUNT; index++) {
        ft_tt_udrv_pci_bar *bar = &out_device->bar[index];
        unsigned long long start = 0;
        unsigned long long end = 0;
        unsigned long long flags = 0;

        if (fgets(line, (int)sizeof(line), file) == NULL) {
            /* A `resource` file with fewer than six BAR lines is not a device
             * we understand. Refuse rather than leave BARs zeroed, which would
             * read as "absent" and silently lose a real aperture. */
            (void)fclose(file);
            return FT_TT_UDRV_ENODEV;
        }

        if (sscanf(line, "%llx %llx %llx", &start, &end, &flags) != 3) {
            (void)fclose(file);
            return FT_TT_UDRV_ENODEV;
        }

        bar->start = (uint64_t)start;
        bar->end = (uint64_t)end;
        bar->flags = (uint64_t)flags;
        classify_bar(bar);
    }

    (void)fclose(file);
    return FT_TT_UDRV_OK;
}

/** Copy the basename of a device directory into `out_address`. */
static int extract_address(const char *device_dir, char *out_address, size_t address_bytes)
{
    const char *slash = strrchr(device_dir, '/');
    const char *base = slash ? (slash + 1) : device_dir;
    size_t length = strlen(base);

    if (length == 0u || length >= address_bytes)
        return FT_TT_UDRV_EINVAL;

    memcpy(out_address, base, length + 1u);
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_pci_read_device(const char *device_dir, ft_tt_udrv_pci_device *out_device)
{
    uint64_t value = 0;
    size_t length;
    int rc;

    if (!device_dir || !out_device)
        return FT_TT_UDRV_EINVAL;

    length = strlen(device_dir);
    if (length == 0u || length >= sizeof(out_device->sysfs_path))
        return FT_TT_UDRV_EINVAL;

    memset(out_device, 0, sizeof(*out_device));
    memcpy(out_device->sysfs_path, device_dir, length + 1u);

    rc = extract_address(device_dir, out_device->address, sizeof(out_device->address));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = read_hex_attribute(device_dir, "vendor", &value);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    out_device->vendor = (uint16_t)value;

    rc = read_hex_attribute(device_dir, "device", &value);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    out_device->device = (uint16_t)value;

    /* Subsystem ids, class and revision are informational — a device that does
     * not publish them is still usable, so they default to 0 rather than
     * failing the read. */
    if (read_hex_attribute(device_dir, "subsystem_vendor", &value) == FT_TT_UDRV_OK)
        out_device->subsystem_vendor = (uint16_t)value;

    if (read_hex_attribute(device_dir, "subsystem_device", &value) == FT_TT_UDRV_OK)
        out_device->subsystem_device = (uint16_t)value;

    if (read_hex_attribute(device_dir, "class", &value) == FT_TT_UDRV_OK)
        out_device->class_code = (uint32_t)value;

    if (read_hex_attribute(device_dir, "revision", &value) == FT_TT_UDRV_OK)
        out_device->revision = (uint8_t)value;

    return read_resource_table(device_dir, out_device);
}

/** qsort comparator over device address, so enumeration order is stable. */
static int compare_by_address(const void *left, const void *right)
{
    const ft_tt_udrv_pci_device *a = left;
    const ft_tt_udrv_pci_device *b = right;

    return strcmp(a->address, b->address);
}

int ft_tt_udrv_pci_find_devices(const char *devices_dir, uint16_t vendor, uint16_t device,
                                ft_tt_udrv_pci_device *out_devices, size_t capacity,
                                size_t *out_count)
{
    const char *root = devices_dir ? devices_dir : FT_TT_UDRV_PCI_DEFAULT_DEVICES_DIR;
    struct dirent *entry;
    DIR *dir;
    size_t found = 0u;

    if (!out_count || (capacity > 0u && !out_devices))
        return FT_TT_UDRV_EINVAL;

    *out_count = 0u;

    dir = opendir(root);
    if (!dir)
        return FT_TT_UDRV_ENODEV;

    while ((entry = readdir(dir)) != NULL) {
        char path[FT_TT_UDRV_PCI_PATH_MAX];
        ft_tt_udrv_pci_device candidate;
        int written;

        if (entry->d_name[0] == '.')
            continue;

        written = snprintf(path, sizeof(path), "%s/%s", root, entry->d_name);
        if (written < 0 || (size_t)written >= sizeof(path))
            continue;

        /* A directory that is not a PCI device, or one we cannot parse, is
         * skipped rather than fatal: /sys/bus/pci/devices holds symlinks to
         * every device on the machine and most are not ours. */
        if (ft_tt_udrv_pci_read_device(path, &candidate) != FT_TT_UDRV_OK)
            continue;

        if (candidate.vendor != vendor)
            continue;

        if (device != 0xFFFFu && candidate.device != device)
            continue;

        if (found < capacity)
            out_devices[found] = candidate;

        found++;
    }

    (void)closedir(dir);

    if (found > 0u && capacity > 0u) {
        size_t sorted = (found < capacity) ? found : capacity;
        qsort(out_devices, sorted, sizeof(*out_devices), compare_by_address);
    }

    *out_count = found;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_pci_resource_path(const ft_tt_udrv_pci_device *device, unsigned bar,
                                 bool write_combine, char *out_path, size_t path_bytes)
{
    int written;

    if (!device || !out_path || bar >= FT_TT_UDRV_PCI_BAR_COUNT)
        return FT_TT_UDRV_EINVAL;

    if (!device->bar[bar].present)
        return FT_TT_UDRV_EINVAL;

    /* The PCI core only creates `resourceN_wc` for a prefetchable BAR, so
     * asking for one on a non-prefetchable BAR would hand back a path that
     * does not exist. Refuse here, where the reason is visible, rather than at
     * an ENOENT three layers up. */
    if (write_combine && !device->bar[bar].prefetchable)
        return FT_TT_UDRV_EINVAL;

    written = snprintf(out_path, path_bytes, "%s/resource%u%s", device->sysfs_path, bar,
                       write_combine ? "_wc" : "");
    if (written < 0 || (size_t)written >= path_bytes)
        return FT_TT_UDRV_EINVAL;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_pci_map_bar(const ft_tt_udrv_pci_device *device, unsigned bar, bool write_combine,
                           bool writable, uint64_t offset, size_t length, void **out_mapping)
{
    char path[FT_TT_UDRV_PCI_PATH_MAX];
    const ft_tt_udrv_pci_bar *description;
    void *mapping;
    long page_size;
    int file;
    int rc;

    if (out_mapping)
        *out_mapping = NULL;

    if (!device || !out_mapping || bar >= FT_TT_UDRV_PCI_BAR_COUNT || length == 0u)
        return FT_TT_UDRV_EINVAL;

    /* Range-check against the BAR the KERNEL published, not against a length
     * the caller believes. A window aimed one register past the end of the
     * register file is not an error anywhere — it is a different register. */
    description = &device->bar[bar];
    if (!description->present)
        return FT_TT_UDRV_EINVAL;

    if (offset > description->length || (uint64_t)length > description->length - offset)
        return FT_TT_UDRV_EINVAL;

    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        return FT_TT_UDRV_EIO;

    /* mmap(2) requires a page-aligned offset, and the PCI core's own
     * pci_mmap_resource() computes a pfn from it — an unaligned offset there
     * would silently map the wrong page rather than fail. */
    if ((offset % (uint64_t)page_size) != 0u)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_pci_resource_path(device, bar, write_combine, path, sizeof(path));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    file = open(path, writable ? (O_RDWR | O_CLOEXEC) : (O_RDONLY | O_CLOEXEC));
    if (file < 0)
        return FT_TT_UDRV_EIO;

    mapping = mmap(NULL, length, writable ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED,
                   file, (off_t)offset);

    /* The mapping outlives the descriptor, so there is nothing to keep open —
     * and an fd we do not hold is an fd a fork cannot leak. */
    (void)close(file);

    if (mapping == MAP_FAILED)
        return FT_TT_UDRV_EIO;

    *out_mapping = mapping;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_pci_unmap_bar(void *mapping, size_t length)
{
    if (!mapping)
        return FT_TT_UDRV_OK;

    if (length == 0u)
        return FT_TT_UDRV_EINVAL;

    return (munmap(mapping, length) == 0) ? FT_TT_UDRV_OK : FT_TT_UDRV_EIO;
}
