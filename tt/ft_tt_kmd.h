/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_kmd.h
 * @brief Foundry's boundary to the Tenstorrent kernel driver's character device
 *   (/dev/tenstorrent/N): open a card, read what it is, claim / map / aim / release TLB
 *   windows, and the two driver side-requests Foundry uses (AI clock, close-time NoC write).
 *
 * Every entry point returns 0 or a NEGATIVE errno unless its comment says otherwise. Each one
 * issues exactly the syscalls its comment lists, once, with no retries — a differential
 * harness compares the syscall sequence and argument bytes, so nothing here may add one.
 *
 * THE NOC FENCE. Aiming a window (CONFIGURE_WINDOW) is the only operation here that issues a
 * NoC transaction, and a wrong one hangs the chip. It exists only in a build that defines
 * FT_TT_PHASE1B_NOC_ENABLED; otherwise ft_tt_kmd_window_aim() returns -EPERM and touches
 * nothing, so a default build cannot reach the NoC at all.
 *
 * @status Complete against the driver ABI in ft_tt_kmd_abi.h. Unit-tested off-device against
 *   fake syscalls (tests/test_kmd.c).
 * @issues None
 * @todo None
 */

#ifndef FT_TT_KMD_H
#define FT_TT_KMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_kmd_abi.h"

/* ================================================================
 * Enums — the VALUES are relied on elsewhere (ft_tt_arch tables, callers' logs)
 * ================================================================ */

/** Chip architecture, as decoded from the PCI device id. */
enum ft_tt_kmd_arch {
    FT_TT_KMD_ARCH_UNKNOWN = 0,
    FT_TT_KMD_ARCH_WORMHOLE = 1,
    FT_TT_KMD_ARCH_BLACKHOLE = 2
};

/** Which mapping of a window's aperture to use: uncached for registers, write-combined for
 *  bulk data. */
enum ft_tt_kmd_cache_mode {
    FT_TT_KMD_CACHE_UC = 0,
    FT_TT_KMD_CACHE_WC = 1
};

/** What ft_tt_kmd_driver_attr() reports. */
enum ft_tt_kmd_driver_attr {
    FT_TT_KMD_DRIVER_API_VERSION = 0,
    FT_TT_KMD_DRIVER_VERSION_MAJOR = 1,
    FT_TT_KMD_DRIVER_VERSION_MINOR = 2,
    FT_TT_KMD_DRIVER_VERSION_PATCH = 3
};

/** NoC ordering of a window's transactions. */
enum ft_tt_kmd_window_ordering {
    FT_TT_KMD_ORDERING_RELAXED = 0,
    FT_TT_KMD_ORDERING_STRICT = 1,
    FT_TT_KMD_ORDERING_POSTED = 2,
    FT_TT_KMD_ORDERING_POSTED_STRICT = 3
};

/* ================================================================
 * Window sizes
 * ================================================================ */

#define FT_TT_KMD_WINDOW_SIZE_1M  (1ull << 20)
#define FT_TT_KMD_WINDOW_SIZE_2M  (2ull << 20)
#define FT_TT_KMD_WINDOW_SIZE_16M (16ull << 20)
#define FT_TT_KMD_WINDOW_SIZE_4G  (4ull << 30)

/** The most 2 MiB windows a Blackhole reports, however large BAR0 is. */
#define FT_TT_KMD_BLACKHOLE_2M_WINDOW_LIMIT 202u

/* ================================================================
 * Types
 * ================================================================ */

/** One open chardev: its fd and the BAR resources cached at open. Opaque. */
typedef struct ft_tt_kmd_device ft_tt_kmd_device;

/** One claimed TLB window: driver id, size and the mapped aperture. Opaque. */
typedef struct ft_tt_kmd_window ft_tt_kmd_window;

/** What the card is (GET_DEVICE_INFO), plus window counts derived from the BAR lengths. */
typedef struct ft_tt_kmd_device_attrs {
    uint64_t pci_domain;
    uint64_t pci_bus;
    uint64_t pci_device;
    uint64_t pci_function;
    uint64_t pci_vendor_id;
    uint64_t pci_device_id;
    uint64_t pci_subsystem_vendor_id;
    uint64_t pci_subsystem_id;
    uint64_t chip_arch;          /**< enum ft_tt_kmd_arch */
    uint64_t window_count_1m;
    uint64_t window_count_2m;
    uint64_t window_count_16m;
    uint64_t window_count_4g;    /**< 0 on a board with no BAR4 */
} ft_tt_kmd_device_attrs;

/** mmap offsets and lengths of the three BARs. A length of 0 means the BAR is absent. */
typedef struct ft_tt_kmd_resources {
    uint64_t bar0_uc_offset;
    uint64_t bar0_wc_offset;
    uint64_t bar0_len;
    uint64_t bar2_uc_offset;
    uint64_t bar2_wc_offset;
    uint64_t bar2_len;
    uint64_t bar4_uc_offset;
    uint64_t bar4_wc_offset;
    uint64_t bar4_len;
} ft_tt_kmd_resources;

/** Where to aim a window. A unicast target has x_end/y_end set and the starts zero. */
typedef struct ft_tt_kmd_window_target {
    uint64_t addr;       /**< tile-local address; a multiple of the window size */
    uint16_t x_end;
    uint16_t y_end;
    uint16_t x_start;
    uint16_t y_start;
    uint8_t noc;
    uint8_t mcast;
    uint8_t ordering;    /**< enum ft_tt_kmd_window_ordering */
    uint8_t static_vc;
} ft_tt_kmd_window_target;

/* ================================================================
 * Device
 * ================================================================ */

/**
 * Open the chardev and cache its BAR resources.
 *
 * Syscalls: open(path, O_RDWR | O_CLOEXEC | extra_flags), then one QUERY_MAPPINGS asking for
 * six records. On an ioctl failure the fd is closed.
 *
 * @param path         e.g. "/dev/tenstorrent/0"
 * @param out_device   receives the handle on success; release with ft_tt_kmd_close()
 * @param extra_flags  added to the open flags (O_APPEND marks a power-aware client)
 * @return 0, -EINVAL (NULL argument), -ENOMEM, or the failing call's -errno
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_kmd_open(const char *path, ft_tt_kmd_device **out_device, int extra_flags);

/**
 * Close the fd and free the handle. NULL is a no-op returning 0. The handle is freed even
 * when close() fails.
 * @return 0 or close()'s -errno
 */
int ft_tt_kmd_close(ft_tt_kmd_device *device);

/**
 * Read the card's identity with one GET_DEVICE_INFO. Window counts are nonzero only on a
 * Blackhole: 2 MiB = BAR0 length / 2 MiB capped at 202, 4 GiB = BAR4 length / 4 GiB.
 * @return 0, -EINVAL, or the ioctl's -errno
 */
int ft_tt_kmd_get_attrs(ft_tt_kmd_device *device, ft_tt_kmd_device_attrs *out_attrs);

/** Copy the BAR resources cached at open. No syscall. @return 0 or -EINVAL */
int ft_tt_kmd_get_resources(const ft_tt_kmd_device *device, ft_tt_kmd_resources *out_resources);

/** The chardev fd, or -1 for NULL. */
int ft_tt_kmd_fd(const ft_tt_kmd_device *device);

/**
 * Ask for the maximum (or minimum) AI clock: one SET_POWER_STATE with every flag valid and
 * the word FT_TT_KMD_POWER_FLAGS_ALL, minus MAX_AI_CLOCK when `want_max` is false.
 * @return 0, -EINVAL, or the ioctl's -errno
 */
int ft_tt_kmd_request_max_ai_clock(ft_tt_kmd_device *device, bool want_max);

/**
 * Register a NoC write the driver performs when this fd closes — including when the process
 * dies. One NOC_CLEANUP ioctl.
 * @param noc    NoC index
 * @param x, y   target tile
 * @param addr   tile-local address
 * @param value  data written there
 * @return 0, -EINVAL, or the ioctl's -errno
 */
int ft_tt_kmd_set_noc_cleanup(ft_tt_kmd_device *device, uint8_t noc, uint8_t x, uint8_t y,
                              uint64_t addr, uint64_t value);

/** Cancel the close-time write: one NOC_CLEANUP with enabled = 0. @return as above */
int ft_tt_kmd_clear_noc_cleanup(ft_tt_kmd_device *device);

/**
 * Report a driver attribute. With a device, one GET_DRIVER_INFO is issued first.
 * FT_TT_KMD_DRIVER_API_VERSION is the compiled-in FT_TT_KMD_ABI_VERSION, answerable with a
 * NULL device. The semver parts come from the driver's reply; with a NULL device they write 0
 * and return -ENODEV.
 * @return 0, -EINVAL (NULL out or unknown attribute), -ENODEV, or the ioctl's -errno
 */
int ft_tt_kmd_driver_attr(ft_tt_kmd_device *device, enum ft_tt_kmd_driver_attr attr,
                          uint64_t *out_value);

/* ================================================================
 * Windows
 * ================================================================ */

/**
 * Claim a window (ALLOCATE_WINDOW) and mmap its aperture with `protection`. If the mmap
 * fails the claim is released with one FREE_WINDOW before returning.
 * @param cache  which of the driver's two aperture offsets to map
 * @return 0, -EINVAL, -ENOMEM, or the failing call's -errno
 */
int ft_tt_kmd_window_alloc(ft_tt_kmd_device *device, size_t size,
                           enum ft_tt_kmd_cache_mode cache, int protection,
                           ft_tt_kmd_window **out_window);

/**
 * munmap the aperture, THEN release the claim (FREE_WINDOW), then free the handle. The order
 * is required: the driver may hand the window to another client the moment it is freed. A
 * failing step is reported but does not stop the later ones.
 * @return 0, -EINVAL, or the last failing call's -errno
 */
int ft_tt_kmd_window_free(ft_tt_kmd_device *device, ft_tt_kmd_window *window);

/** The mapped aperture. @return 0 or -EINVAL */
int ft_tt_kmd_window_aperture(const ft_tt_kmd_window *window, void **out_aperture);

/** The driver-assigned window id. @return 0 or -EINVAL */
int ft_tt_kmd_window_id(const ft_tt_kmd_window *window, uint32_t *out_id);

/** The window size in bytes. @return 0 or -EINVAL */
int ft_tt_kmd_window_size(const ft_tt_kmd_window *window, size_t *out_size);

/**
 * Aim a window (CONFIGURE_WINDOW). FENCED: without FT_TT_PHASE1B_NOC_ENABLED this returns
 * -EPERM and touches nothing.
 * @return 0, -EPERM (fence), -EINVAL (NULL argument, or addr not a multiple of the window
 *         size), or the ioctl's -errno
 */
int ft_tt_kmd_window_aim(ft_tt_kmd_device *device, ft_tt_kmd_window *window,
                         const ft_tt_kmd_window_target *target);

/** Aim a window at one tile: a zeroed target with addr, x_end = x, y_end = y. Fenced the
 *  same way as ft_tt_kmd_window_aim(). */
int ft_tt_kmd_window_aim_unicast(ft_tt_kmd_device *device, ft_tt_kmd_window *window,
                                 uint16_t x, uint16_t y, uint64_t addr);

#endif /* FT_TT_KMD_H */
