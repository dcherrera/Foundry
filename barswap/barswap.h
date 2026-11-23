/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * barswap.h — Userspace library for BAR Swap kernel module v2
 *
 * Provides a simple C API to query GPU die state, and dynamically
 * acquire/release individual K80 dies into the single MMIO slot.
 * Link with barswap.c — no external dependencies.
 */

#ifndef FOUNDRY_BARSWAP_H
#define FOUNDRY_BARSWAP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Max dies/slots (mirrors kernel module limits) */
#define BS_MAX_DIES  16
#define BS_MAX_SLOTS 8

/* Opaque context — holds fd + cached state */
typedef struct bs_context bs_context;

/*
 * bs_gpu — Handle to an acquired GPU die
 *
 * Returned by bs_acquire(). Contains the physical addresses and
 * an mmap'd pointer to VRAM. Release with bs_release().
 */
typedef struct bs_gpu {
    int die_id;
    uint64_t bar1_addr;         /* VRAM physical address */
    uint64_t bar1_size;         /* 16 GB */
    void *vram;                 /* mmap'd VRAM pointer (or NULL if mmap failed) */
    size_t vram_mapped_size;
    int mem_fd;                 /* /dev/mem fd */
} bs_gpu;

/*
 * bs_init — Open /dev/bar_swap and query initial state.
 * Returns NULL on failure (e.g., device not present).
 */
bs_context* bs_init(void);

/*
 * bs_free — Close device and free context.
 */
void bs_free(bs_context *ctx);

/*
 * bs_num_dies — Total number of GPU dies discovered by the kernel module.
 */
uint32_t bs_num_dies(const bs_context *ctx);

/*
 * bs_num_slots — Number of MMIO slots configured (always 1 in v2).
 */
uint32_t bs_num_slots(const bs_context *ctx);

/*
 * bs_is_mapped — Returns 1 if die_id has BARs programmed, 0 otherwise.
 * Returns 0 for out-of-range die_id.
 */
int bs_is_mapped(const bs_context *ctx, int die_id);

/*
 * bs_acquire — Acquire a die into the MMIO slot.
 *
 * Issues ACQUIRE ioctl, then opens /dev/mem and mmaps VRAM (BAR 1).
 * If a different die is currently active, it is released implicitly.
 * Returns a bs_gpu handle on success, NULL on failure.
 */
bs_gpu* bs_acquire(bs_context *ctx, int die_id);

/*
 * bs_release — Release a die from the MMIO slot.
 *
 * Unmaps VRAM, closes /dev/mem, issues RELEASE ioctl, frees the handle.
 * Safe to call with NULL.
 */
void bs_release(bs_context *ctx, bs_gpu *gpu);

/*
 * bs_active_die — Returns the die currently in the MMIO slot, or -1 if empty.
 *
 * Queries the kernel module for current state.
 */
int bs_active_die(bs_context *ctx);

/*
 * bs_refresh — Re-query the kernel module and update cached state.
 *
 * Returns 0 on success, -1 on failure.
 */
int bs_refresh(bs_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_BARSWAP_H */
