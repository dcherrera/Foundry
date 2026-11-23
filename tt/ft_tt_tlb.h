/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tlb.h
 * @brief TLB window ownership for the Foundry Blackhole backend: claim a
 *   window from the driver, map its aperture, hand back the register offset
 *   Phase 1b will slide it with, and release it.
 *
 * A TLB window is the ONLY host<->card data path on this board. There is no
 * BAR4 and therefore no 4 GiB window (spec §2.3): 202 x 2 MiB windows exist,
 * the driver keeps index 201, and 201 are ours. A window is a *sliding*
 * aperture — moving it means rewriting three 32-bit registers in BAR0,
 * not re-mapping anything.
 *
 * This file covers ownership ONLY. Aiming a window (CONFIGURE_TLB) and moving
 * data through it are Phase 1b, and are fenced off at the kernel boundary
 * (ft_tt_kmd.h, "THE NOC FENCE"). The split is deliberate: allocation is pure driver
 * bookkeeping that cannot hurt the card, aiming is the operation that hung it.
 *
 * @status Phase 1a. Allocate / query / free have run against the p150a on the
 *   bank, including exhausting the pool to confirm the usable count. The
 *   aperture is mapped PROT_NONE, so it cannot be dereferenced at all.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_TLB_H
#define FT_TT_TLB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_device.h"
#include "ft_tt_kmd.h"

/** One claimed TLB window. Opaque — see ft_tt_tlb.c. */
typedef struct ft_tt_tlb_window ft_tt_tlb_window;

/**
 * Claim a TLB window and map its aperture.
 *
 * The claim is a bit in the driver's per-device bitmap, shared across every
 * process that has the card open. It is the arbitration
 * mechanism against both the kernel's own window and any other client, and it
 * touches no hardware.
 *
 * The aperture is mapped **PROT_NONE** in Phase 1a: an unaimed window points
 * at an undefined NoC endpoint, and PROT_NONE makes dereferencing it a SIGSEGV
 * on the host instead of a transaction on the chip. Building with
 * FT_TT_PHASE1B_NOC_ENABLED maps it PROT_READ|PROT_WRITE instead.
 *
 * @param ctx    open device context
 * @param size   FT_TT_KMD_WINDOW_SIZE_2M — the only size this board has
 * @param cache  UC for register access, WC for bulk data
 * @param out_window  receives the window; release with ft_tt_tlb_window_free()
 * @return       FT_TT_OK; FT_TT_EINVAL for a bad argument or a window size this
 *               board does not have; FT_TT_ENOMEM when every window is in use;
 *               FT_TT_EIO otherwise. The failing errno is logged.
 *
 * Thread-safe: yes for the claim; the context's window tally is not locked
 * Reentrant: yes
 * Signal-safe: no
 */
int ft_tt_tlb_window_alloc(ft_tt_ctx *ctx, size_t size, enum ft_tt_kmd_cache_mode cache,
                           ft_tt_tlb_window **out_window);

/**
 * Unmap the aperture, release the claim, free the handle. In that order — the
 * driver requires the munmap first.
 *
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_EIO (logged).
 *
 * Thread-safe: no (frees `window`)    Reentrant: no    Signal-safe: no
 */
int ft_tt_tlb_window_free(ft_tt_ctx *ctx, ft_tt_tlb_window *window);

/** Driver-assigned window index, or UINT32_MAX for a NULL window. */
uint32_t ft_tt_tlb_window_id(const ft_tt_tlb_window *window);

/** Window size in bytes, or 0. */
size_t ft_tt_tlb_window_size(const ft_tt_tlb_window *window);

/**
 * Byte offset of this window's three configuration registers within the TLB
 * register file returned by ft_tt_tlb_registers() — `id * 12`. This is what Phase 1b writes to slide the
 * window without an ioctl.
 *
 * @return the offset, or SIZE_MAX for a NULL window.
 */
size_t ft_tt_tlb_window_register_offset(const ft_tt_tlb_window *window);

/**
 * The mapped aperture.
 *
 * DEREFERENCING THIS IS A NoC TRANSACTION to wherever the window is aimed, and
 * in Phase 1a it is aimed nowhere and mapped PROT_NONE. The accessor exists so
 * Phase 1b has the pointer ready; Phase 1a must not touch it.
 *
 * @return the aperture, or NULL.
 */
volatile void *ft_tt_tlb_window_aperture(const ft_tt_tlb_window *window);

/**
 * Whether this window has ever been aimed at a NoC endpoint. Always false in
 * Phase 1a — nothing here calls CONFIGURE_TLB. Phase 1b sets it, and the data
 * path can then refuse to touch an aperture that was never aimed.
 */
bool ft_tt_tlb_window_is_aimed(const ft_tt_tlb_window *window);

/**
 * Aim a window at a unicast NoC endpoint (CONFIGURE_TLB).
 *
 * **PHASE 1B.** With the default build this returns FT_TT_EPERM without
 * touching the device — the fence is in ft_tt_kmd.c. It is the one
 * operation in this file that can hang the chip, which is why it is separated
 * from allocation.
 *
 * A coordinate must come from ARC telemetry or from a source the driver has
 * already validated. NEVER pass a probed or guessed coordinate.
 *
 * @param x, y  NoC0 coordinates of the target tile
 * @param addr  tile-local address, aligned to the window size
 * @return      FT_TT_OK; FT_TT_EPERM when the fence is closed; FT_TT_EINVAL for
 *              a misaligned address; FT_TT_EIO otherwise.
 *
 * Thread-safe: no (mutates the window)    Reentrant: no    Signal-safe: no
 */
int ft_tt_tlb_window_aim(ft_tt_ctx *ctx, ft_tt_tlb_window *window, uint16_t x, uint16_t y,
                         uint64_t addr);

/** The tile-local address this window is currently aimed at, or 0 if it has
 *  never been aimed. Phase 1b turns a NoC address into an aperture offset with
 *  `noc_addr - ft_tt_tlb_window_base_address(window)`. */
uint64_t ft_tt_tlb_window_base_address(const ft_tt_tlb_window *window);

#endif /* FT_TT_TLB_H */
