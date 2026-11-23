/* SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * bar_swap.h — Shared ioctl definitions for BAR Swap kernel module
 *
 * Used by both the kernel module (bar_swap.ko) and userspace library (barswap.c).
 * Provides ioctl commands to query GPU die/slot mapping state and
 * acquire/release individual dies into the single MMIO slot.
 */

#ifndef BAR_SWAP_H
#define BAR_SWAP_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <sys/ioctl.h>
#include <stdint.h>
#ifdef __linux__
#include <linux/types.h>   /* glibc's fcntl.h may pull it in too; never redefine */
#else
typedef uint32_t __u32;
typedef uint64_t __u64;
#endif
#endif

#define BAR_SWAP_MAGIC 'B'

/* Maximum number of GPU dies we track */
#define BS_MAX_DIES  16

/* Maximum number of MMIO slots available */
#define BS_MAX_SLOTS 8

/*
 * bs_query_info — returned by BAR_SWAP_QUERY ioctl
 *
 * n_dies:      total number of K80 GPU dies discovered
 * n_slots:     number of MMIO slots configured (always 1 in v2)
 * mapped:      mapped[i] = 1 if die i has BARs programmed into the MMIO slot
 * active_die:  die currently occupying the slot, or 0xFFFFFFFF if none
 */
struct bs_query_info {
    __u32 n_dies;
    __u32 n_slots;
    __u32 mapped[BS_MAX_DIES];
    __u32 active_die;
};

/*
 * bs_acquire_req — sent to BAR_SWAP_ACQUIRE ioctl
 *
 * Programs the requested die into the single MMIO slot. If a different die
 * is currently active, it is released first (implicit release).
 *
 * die_id:    which die to acquire (in)
 * flags:     BS_ACQUIRE_FORCE_SBR to force a Secondary Bus Reset (in)
 * bar0_addr: programmed BAR 0 physical address (out)
 * bar1_addr: programmed BAR 1 (VRAM) physical address (out)
 * bar1_size: BAR 1 size in bytes (out)
 * bar3_addr: programmed BAR 3 physical address (out)
 */
struct bs_acquire_req {
    __u32 die_id;
    __u32 flags;
    __u64 bar0_addr;
    __u64 bar1_addr;
    __u64 bar1_size;
    __u64 bar3_addr;
};

/*
 * bs_release_req — sent to BAR_SWAP_RELEASE ioctl
 *
 * Releases the specified die from the MMIO slot. Disables decode,
 * releases resources, zeros BARs.
 *
 * die_id: which die to release (in)
 * flags:  reserved, must be 0 (in)
 */
struct bs_release_req {
    __u32 die_id;
    __u32 flags;
};

/* Acquire flags */
#define BS_ACQUIRE_FORCE_SBR  0x1

/* Query current die/slot mapping state */
#define BAR_SWAP_QUERY    _IOR(BAR_SWAP_MAGIC, 3, struct bs_query_info)

/* Acquire a die into the MMIO slot */
#define BAR_SWAP_ACQUIRE  _IOWR(BAR_SWAP_MAGIC, 4, struct bs_acquire_req)

/* Release a die from the MMIO slot */
#define BAR_SWAP_RELEASE  _IOW(BAR_SWAP_MAGIC, 5, struct bs_release_req)

#endif /* BAR_SWAP_H */
