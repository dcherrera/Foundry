/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tlb.c
 * @brief Implementation of TLB window ownership — ALLOCATE_TLB, the aperture
 *   mmap, and FREE_TLB. No window is ever aimed here; that is Phase 1b.
 *
 * @status Phase 1a, exercised on the p150a test card.
 * @issues None
 * @todo None
 */

/* POSIX.1-2008 is needed for mmap's PROT_* under -std=c99, which turns off the
 * implicit feature-test macros glibc would otherwise define. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_tlb.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>

struct ft_tt_tlb_window {
    ft_tt_kmd_window *kmd_window;
    uint32_t id;
    size_t size;
    void *aperture;
    bool aimed;
    uint64_t base_address;
};

/*
 * Phase 1a maps the aperture PROT_NONE: the window is claimed but aimed
 * nowhere, so a stray dereference must fault on the host rather than issue a
 * NoC transaction to an undefined endpoint (spec §7.3 — that mistake costs a
 * physical power cycle). Phase 1b flips this to a real mapping.
 */
static int aperture_protection(void)
{
#ifdef FT_TT_PHASE1B_NOC_ENABLED
    return PROT_READ | PROT_WRITE;
#else
    return PROT_NONE;
#endif
}

/* The kernel boundary speaks errno; the Foundry surface speaks FT_TT_*. */
static int status_from_errno(int negative_errno)
{
    switch (-negative_errno) {
    case EINVAL:
        return FT_TT_EINVAL;
    case ENOMEM:
        return FT_TT_ENOMEM;
    case EPERM:
        return FT_TT_EPERM;
    case ENODEV:
        return FT_TT_ENODEV;
    default:
        return FT_TT_EIO;
    }
}

int ft_tt_tlb_window_alloc(ft_tt_ctx *ctx, size_t size, enum ft_tt_kmd_cache_mode cache,
                           ft_tt_tlb_window **out_window)
{
    ft_tt_tlb_window *window = NULL;
    int rc;

    if (!ctx || !out_window || !ft_tt_ctx_available(ctx))
        return FT_TT_EINVAL;

    window = calloc(1, sizeof(*window));
    if (!window) {
        ft_tt_log_error("out of memory allocating a TLB window handle");
        return FT_TT_ENOMEM;
    }

    rc = ft_tt_kmd_window_alloc(ft_tt_ctx_kmd_device(ctx), size, cache, aperture_protection(),
                                &window->kmd_window);
    if (rc != 0) {
        /* -EINVAL here means the board has no window of this size at all, which
         * is exactly what a 4 GiB request gets with no BAR4 (spec §2.3). */
        ft_tt_log_error("ALLOCATE_TLB(%zu bytes) failed on device %d: %s", size,
                        ft_tt_ctx_device_index(ctx), strerror(-rc));
        free(window);
        return status_from_errno(rc);
    }

    if (ft_tt_kmd_window_id(window->kmd_window, &window->id) != 0 ||
        ft_tt_kmd_window_size(window->kmd_window, &window->size) != 0 ||
        ft_tt_kmd_window_aperture(window->kmd_window, &window->aperture) != 0) {
        ft_tt_log_error("TLB window accessors failed after a successful allocation");
        (void)ft_tt_kmd_window_free(ft_tt_ctx_kmd_device(ctx), window->kmd_window);
        free(window);
        return FT_TT_ERROR;
    }

    ft_tt_ctx_note_window(ctx, +1);
    *out_window = window;
    return FT_TT_OK;
}

int ft_tt_tlb_window_free(ft_tt_ctx *ctx, ft_tt_tlb_window *window)
{
    int rc;

    if (!ctx || !window)
        return FT_TT_EINVAL;

    rc = ft_tt_kmd_window_free(ft_tt_ctx_kmd_device(ctx), window->kmd_window);
    if (rc != 0)
        ft_tt_log_error("FREE_TLB(%u) failed on device %d: %s", window->id,
                        ft_tt_ctx_device_index(ctx), strerror(-rc));

    ft_tt_ctx_note_window(ctx, -1);
    free(window);

    return rc == 0 ? FT_TT_OK : status_from_errno(rc);
}

uint32_t ft_tt_tlb_window_id(const ft_tt_tlb_window *window)
{
    return window ? window->id : UINT32_MAX;
}

size_t ft_tt_tlb_window_size(const ft_tt_tlb_window *window)
{
    return window ? window->size : 0;
}

size_t ft_tt_tlb_window_register_offset(const ft_tt_tlb_window *window)
{
    return window ? (size_t)window->id * FT_TT_TLB_REG_STRIDE : SIZE_MAX;
}

volatile void *ft_tt_tlb_window_aperture(const ft_tt_tlb_window *window)
{
    return window ? window->aperture : NULL;
}

bool ft_tt_tlb_window_is_aimed(const ft_tt_tlb_window *window)
{
    return window ? window->aimed : false;
}

int ft_tt_tlb_window_aim(ft_tt_ctx *ctx, ft_tt_tlb_window *window, uint16_t x, uint16_t y,
                         uint64_t addr)
{
    int rc;

    if (!ctx || !window)
        return FT_TT_EINVAL;

    /* Phase 1a: ft_tt_kmd_window_aim_unicast returns -EPERM without issuing the
     * ioctl, so the device is never reached. Phase 1b builds with
     * FT_TT_PHASE1B_NOC_ENABLED and this becomes a real CONFIGURE_WINDOW. */
    rc = ft_tt_kmd_window_aim_unicast(ft_tt_ctx_kmd_device(ctx), window->kmd_window, x, y, addr);
    if (rc != 0) {
        ft_tt_log_error("aiming window %u at NoC0 (%u, %u) 0x%llx failed: %s", window->id,
                        (unsigned)x, (unsigned)y, (unsigned long long)addr, strerror(-rc));
        return status_from_errno(rc);
    }

    window->aimed = true;
    window->base_address = addr;
    return FT_TT_OK;
}

uint64_t ft_tt_tlb_window_base_address(const ft_tt_tlb_window *window)
{
    return window ? window->base_address : 0;
}
