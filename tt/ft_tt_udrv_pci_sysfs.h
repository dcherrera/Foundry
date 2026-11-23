/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_pci_sysfs.h
 * @brief Find the card and describe its BARs using nothing but sysfs — the
 *   replacement for tt-kmd's GET_DEVICE_INFO and QUERY_MAPPINGS ioctls.
 *
 *   Everything those two ioctls report is already published by the PCI core:
 *   `vendor`, `device`, `subsystem_vendor`, `subsystem_device`, `revision`,
 *   `class` are one-line files, and `resource` is the BAR table
 *   `pci_resource_start/end/flags` verbatim. QUERY_MAPPINGS adds only the mmap
 *   COOKIES the driver invented for its own character device, which a driver
 *   that mmaps `resource0_wc` directly does not need.
 *
 *   The parsing is the whole risk and it is fully testable off-hardware: this
 *   module takes the sysfs ROOT as a parameter, so a test can point it at a
 *   scratch directory containing the same files. `tests/test_udrv_pci_sysfs.c`
 *   does exactly that, including the cases a real card cannot produce on
 *   demand — an absent BAR, a non-prefetchable BAR, a truncated `resource`,
 *   a device with the wrong ids.
 *
 *   WHAT A FIXTURE CANNOT TEST, stated so nobody assumes otherwise: that the
 *   numbers sysfs prints are the ones the hardware decodes; that
 *   `resource0_wc` can actually be mmap'd (it needs root or a udev rule, and
 *   it needs `iomem_is_exclusive()` to say no — it does here, because tt-kmd
 *   never calls `pci_request_regions` and `CONFIG_IO_STRICT_DEVMEM` is not
 *   set on this kernel); and that a write-combined store to that mapping
 *   reaches the card. `tests/test_udrv_card_tlb.c` checks all three ON THE
 *   CARD, in that order, and is the only thing here that opens a device.
 *
 * @status Discovery and parsing are gated against a synthetic sysfs tree.
 *   `ft_tt_udrv_pci_map_bar()` is implemented as of 2026-08-16 and has mapped
 *   BAR0 of the live p150a — read-only first, then writable — through
 *   `resource0` and `resource0_wc`. It needs root, or the udev rule in this
 *   file's notes.
 * @issues
 *   - The BAR files are mode 0600 root:root, so every caller needs root or a
 *     udev rule. **THE RULE IS WRITTEN AND EXERCISED (2026-08-18):**
 *     `tools/60-foundry-blackhole.rules`, gated by
 *     `tests/run_udev_nonroot_gate.sh`. With it installed and `tenstorrent`
 *     unloaded, an ordinary user in the `tenstorrent` group mapped both BARs,
 *     aimed windows, booted firmware on a core and byte-compared a `vector_add`
 *     — no sudo anywhere. Red arm: the BARs back at 0600 make the same command
 *     exit 1 with this file's own message. A user outside the group is still
 *     refused.
 *   - **A udev rule cannot grant CONFIG SPACE**, and that is where the privilege
 *     line really sits: an unprivileged read of `config` is truncated by the
 *     kernel to the 64-byte standard header, so MRRS and the ASIC reset stay
 *     root-only. The rule performs MRRS itself with `setpci` at device-add time
 *     for exactly that reason.
 * @todo None
 */

#ifndef FT_TT_UDRV_PCI_SYSFS_H
#define FT_TT_UDRV_PCI_SYSFS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** PCI ids. Same constants ft_tt_kmd_abi.h carries, restated here so this
 *  module does not include the layer it replaces. */
#define FT_TT_UDRV_PCI_VENDOR_TENSTORRENT 0x1e52u
#define FT_TT_UDRV_PCI_DEVICE_BLACKHOLE 0xb140u

/** BARs sysfs publishes in `resource`, before the ROM and bridge windows. */
#define FT_TT_UDRV_PCI_BAR_COUNT 6u

/** Longest sysfs path this module builds, including the terminator. */
#define FT_TT_UDRV_PCI_PATH_MAX 320u

/** Longest "0000:12:00.0"-shaped address, including the terminator. */
#define FT_TT_UDRV_PCI_ADDRESS_MAX 16u

/** Default place to look for devices. */
#define FT_TT_UDRV_PCI_DEFAULT_DEVICES_DIR "/sys/bus/pci/devices"

/**
 * One BAR, exactly as `resource` reports it.
 *
 * `length` follows the kernel's own `pci_resource_len()`: zero when `end` is
 * zero, otherwise `end - start + 1`. That is not the same as `end >= start`,
 * and the difference is the absent-BAR case — this board has no BAR4 and its
 * line reads all zeros.
 */
typedef struct ft_tt_udrv_pci_bar {
    uint64_t start;
    uint64_t end;
    uint64_t flags;
    uint64_t length;   /**< 0 means the BAR is absent */
    bool present;      /**< length > 0 */
    bool memory;       /**< IORESOURCE_MEM */
    bool io;           /**< IORESOURCE_IO */
    bool prefetchable; /**< IORESOURCE_PREFETCH — also decides whether sysfs
                        *   publishes a `resourceN_wc` companion at all */
    bool is_64bit;     /**< IORESOURCE_MEM_64 */
} ft_tt_udrv_pci_bar;

/** A PCI device as sysfs describes it. */
typedef struct ft_tt_udrv_pci_device {
    char sysfs_path[FT_TT_UDRV_PCI_PATH_MAX]; /**< the device directory */
    char address[FT_TT_UDRV_PCI_ADDRESS_MAX]; /**< its basename, e.g. 0000:12:00.0 */
    uint16_t vendor;
    uint16_t device;
    uint16_t subsystem_vendor;
    uint16_t subsystem_device;
    uint32_t class_code;
    uint8_t revision;
    ft_tt_udrv_pci_bar bar[FT_TT_UDRV_PCI_BAR_COUNT];
} ft_tt_udrv_pci_device;

/**
 * Read one device directory.
 *
 * @param device_dir  e.g. "/sys/bus/pci/devices/0000:12:00.0". The basename
 *                    becomes `address`
 * @param out_device  receives the description
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL on a NULL/oversized
 *                    argument; FT_TT_UDRV_ENODEV when a required attribute is
 *                    missing or unparseable
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_pci_read_device(const char *device_dir, ft_tt_udrv_pci_device *out_device);

/**
 * Enumerate every device under `devices_dir` matching `vendor`/`device`, in
 * ascending address order so device 0 is stable across boots.
 *
 * @param devices_dir  NULL for FT_TT_UDRV_PCI_DEFAULT_DEVICES_DIR
 * @param vendor       PCI vendor id to match
 * @param device       PCI device id to match, or 0xFFFF for "any"
 * @param out_devices  receives up to `capacity` matches
 * @param capacity     size of `out_devices`
 * @param out_count    receives how many matched. May exceed `capacity`, in
 *                     which case only `capacity` were written and the return
 *                     is still FT_TT_UDRV_OK — the caller decides whether that
 *                     is an error
 * @return             FT_TT_UDRV_OK, FT_TT_UDRV_EINVAL, or FT_TT_UDRV_ENODEV
 *                     when `devices_dir` cannot be read
 */
int ft_tt_udrv_pci_find_devices(const char *devices_dir, uint16_t vendor, uint16_t device,
                                ft_tt_udrv_pci_device *out_devices, size_t capacity,
                                size_t *out_count);

/**
 * Build the sysfs filename for a BAR's mapping file: `resourceN`, or
 * `resourceN_wc` for the write-combined companion.
 *
 * The `_wc` file only EXISTS when the BAR is prefetchable — the PCI core
 * creates it conditionally. BAR0 on this board is 512 MiB prefetchable, which
 * is why the write-combined upload path is reachable at all.
 *
 * CORRECTION, checked on the card 2026-08-16: BAR2 here is prefetchable TOO
 * (`resource2_wc` exists on 0000:12:00.0), so the earlier note that it is
 * uncached-only was wrong. It does not change any decision — registers want
 * uncached and get it through `resourceN` — but the reason must be "we chose
 * UC", not "the file does not exist".
 *
 * @return FT_TT_UDRV_OK, or FT_TT_UDRV_EINVAL (bad bar index / no room /
 *         `write_combine` asked for on a non-prefetchable BAR)
 */
int ft_tt_udrv_pci_resource_path(const ft_tt_udrv_pci_device *device, unsigned bar,
                                 bool write_combine, char *out_path, size_t path_bytes);

/**
 * Map `length` bytes of a BAR at `offset` into this process — the whole of the
 * host-side work that replaces tt-kmd's mmap path.
 *
 * `open()` the path `ft_tt_udrv_pci_resource_path()` produced and `mmap()` it.
 * For a TLB window pass `offset = ft_tt_udrv_tlb2m_aperture_offset(id)` and
 * `length = FT_TT_UDRV_TLB_2M_WINDOW_SIZE` with `write_combine = true`: that
 * is bit for bit what tt-kmd's `map_tlb_window()` does
 * (`io_remap_pfn_range(bar_start + id * 2 MiB)` with `pgprot_writecombine`),
 * with the ioctl removed. For the TLB REGISTER file pass
 * `offset = FT_TT_UDRV_TLB_REGS_START`, `length = FT_TT_UDRV_TLB_REGS_LEN` and
 * `write_combine = false` — registers are uncached, which is what the driver's
 * own `pci_iomap_range()` gives it.
 *
 * `writable` is a SAFETY parameter, not a convenience. The read-only rung of
 * bring-up must be mechanically incapable of writing a register, not merely
 * careful about it: a read-only mapping makes a stray store a SIGSEGV on the
 * host instead of a wrong aim on the card. Pass false unless the caller's
 * whole purpose is to write.
 *
 * @param device        the device whose BAR to map
 * @param bar           BAR index; must be present
 * @param write_combine map through `resourceN_wc`; requires a prefetchable BAR
 * @param writable      PROT_READ|PROT_WRITE and O_RDWR when true, else
 *                      PROT_READ and O_RDONLY
 * @param offset        byte offset into the BAR; must be page-aligned, and
 *                      offset + length must fit the BAR
 * @param length        bytes to map; must be non-zero
 * @param out_mapping   receives the mapping; release with
 *                      ft_tt_udrv_pci_unmap_bar(mapping, length)
 * @return              FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL for a bad argument, a
 *                      misaligned offset or a range past the end of the BAR;
 *                      FT_TT_UDRV_EIO when open() or mmap() fails (errno is
 *                      left set, and EACCES means "not root and no udev rule")
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_pci_map_bar(const ft_tt_udrv_pci_device *device, unsigned bar, bool write_combine,
                           bool writable, uint64_t offset, size_t length, void **out_mapping);

/**
 * Release a mapping from ft_tt_udrv_pci_map_bar().
 *
 * @param mapping  what map_bar returned; NULL is a no-op success
 * @param length   the same length map_bar was given
 * @return         FT_TT_UDRV_OK, FT_TT_UDRV_EINVAL, or FT_TT_UDRV_EIO
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_pci_unmap_bar(void *mapping, size_t length);

#endif /* FT_TT_UDRV_PCI_SYSFS_H */
