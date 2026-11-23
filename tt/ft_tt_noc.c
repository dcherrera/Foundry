/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_noc.c
 * @brief Implementation of the named NoC read/write primitives: alignment
 *   rules, the SSE2-shaped PIO copy, the 32-bit register copy, and the window
 *   walk that makes a transfer larger than one 2 MiB aperture look like a
 *   single call. This is the ONLY window walk in the backend — ft_tt_launch's
 *   run control comes through ft_tt_noc_transfer_raw().
 *
 * @status Written on the Mac, verified on the card — docs/tt_bringup_log.md §9
 *   has the step-by-step results and the nine mutation proofs, two of which
 *   were run against silicon. The WORD32 half moved here from ft_tt_launch.c
 *   unchanged in shape (one volatile 32-bit access per word).
 * @issues None
 * @todo None
 */

#include "ft_tt_noc.h"

#include <string.h>

/* ------------------------------------------------------------------------- */
/* Alignment and bounds                                                      */
/* ------------------------------------------------------------------------- */

const char *ft_tt_memory_space_string(ft_tt_memory_space space)
{
    switch (space) {
    case FT_TT_MEM_SPACE_L1:
        return "L1";
    case FT_TT_MEM_SPACE_DRAM:
        return "DRAM";
    default:
        return "?";
    }
}

int ft_tt_noc_check_raw_transfer(ft_tt_noc_transfer_width width, uint64_t address, size_t bytes)
{
    uint32_t unit = ft_tt_noc_transfer_width_bytes(width);

    if (unit == 0u) {
        ft_tt_log_error("noc: unknown transfer width %d", (int)width);
        return FT_TT_EINVAL;
    }
    if ((address % unit) != 0u) {
        ft_tt_log_error("noc: address 0x%llx is not %u-byte aligned",
                        (unsigned long long)address, unit);
        return FT_TT_EINVAL;
    }
    if ((bytes % unit) != 0u) {
        ft_tt_log_error("noc: length %zu is not a multiple of %u", bytes, unit);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

uint32_t ft_tt_noc_alignment(ft_tt_memory_space space, bool is_read)
{
    switch (space) {
    case FT_TT_MEM_SPACE_L1:
        return FT_TT_L1_ALIGNMENT;
    case FT_TT_MEM_SPACE_DRAM:
        /* Writes to DRAM take the L1 alignment; only the read direction needs
         * the full 64 B (ft_tt_noc.h). */
        return is_read ? FT_TT_DRAM_ALIGNMENT : FT_TT_L1_ALIGNMENT;
    default:
        return 0u;
    }
}

uint64_t ft_tt_noc_space_size(ft_tt_memory_space space)
{
    switch (space) {
    case FT_TT_MEM_SPACE_L1:
        return (uint64_t)FT_TT_MEM_L1_SIZE;
    case FT_TT_MEM_SPACE_DRAM:
        return (uint64_t)FT_TT_MEM_DRAM_SIZE;
    default:
        return 0u;
    }
}

int ft_tt_noc_check_transfer(ft_tt_memory_space space, bool is_read, uint64_t address, size_t bytes)
{
    uint32_t alignment = ft_tt_noc_alignment(space, is_read);
    uint64_t limit = ft_tt_noc_space_size(space);

    if (alignment == 0u || limit == 0u) {
        ft_tt_log_error("noc: unknown memory space %d", (int)space);
        return FT_TT_EINVAL;
    }
    if ((address % alignment) != 0u) {
        ft_tt_log_error("noc: %s %s address 0x%llx is not %u-byte aligned",
                        ft_tt_memory_space_string(space), is_read ? "read" : "write",
                        (unsigned long long)address, alignment);
        return FT_TT_EINVAL;
    }
    if ((bytes % alignment) != 0u) {
        ft_tt_log_error("noc: %s %s length %zu is not a multiple of %u",
                        ft_tt_memory_space_string(space), is_read ? "read" : "write", bytes,
                        alignment);
        return FT_TT_EINVAL;
    }
    if ((uint64_t)bytes > limit || address > limit - (uint64_t)bytes) {
        ft_tt_log_error("noc: %s span 0x%llx + %zu runs past the %llu-byte space",
                        ft_tt_memory_space_string(space), (unsigned long long)address, bytes,
                        (unsigned long long)limit);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

int ft_tt_noc_check_transfer_pair(ft_tt_memory_space local_space, uint64_t local_address,
                                  ft_tt_memory_space remote_space, uint64_t remote_address,
                                  bool is_read, size_t bytes)
{
    /* The local end is written when the transfer READS the remote one, and vice versa —
     * which is what decides whether each end gets the 16-byte write rule or the 64-byte
     * read rule on DRAM. */
    uint32_t local_alignment = ft_tt_noc_alignment(local_space, !is_read);
    uint32_t remote_alignment = ft_tt_noc_alignment(remote_space, is_read);
    uint32_t window;
    int rc;

    rc = ft_tt_noc_check_transfer(local_space, !is_read, local_address, bytes);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_noc_check_transfer(remote_space, is_read, remote_address, bytes);
    if (rc != FT_TT_OK)
        return rc;

    /* THE CHECK NEITHER END'S OWN TEST CAN MAKE. Both ends have passed their own rule by
     * here; what is left is that they agree with EACH OTHER inside the stricter window. */
    window = (local_alignment > remote_alignment) ? local_alignment : remote_alignment;
    if (window == 0u)
        return FT_TT_EINVAL;
    if ((local_address & (window - 1u)) != (remote_address & (window - 1u))) {
        ft_tt_log_error("noc: %s 0x%llx and %s 0x%llx sit at different offsets inside a "
                        "%u-byte window; a transfer needs its two ends congruent, not merely "
                        "each aligned",
                        ft_tt_memory_space_string(local_space),
                        (unsigned long long)local_address,
                        ft_tt_memory_space_string(remote_space),
                        (unsigned long long)remote_address, window);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- *
 * The PIO copy lives in ft_tt_noc_copy.h — copy_unit, copy_blocked,
 * copy_words and copy_slice moved there UNCHANGED so that the userspace-driver
 * transport issues byte-for-byte the same store shape. `bytes` is always a
 * multiple of FT_TT_NOC_COPY_UNIT_BYTES here: ft_tt_noc_check_transfer()
 * enforces at least 16 B alignment on both address and length, and the window
 * walk splits only on 2 MiB boundaries, which are 16 B multiples.
 * ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- */
/* Path lifecycle                                                            */
/* ------------------------------------------------------------------------- */

int ft_tt_noc_path_attach(ft_tt_ctx *ctx, ft_tt_tlb_window *window, ft_tt_noc_path *path)
{
    if (!ctx || !window || !path || !ft_tt_ctx_available(ctx))
        return FT_TT_EINVAL;

    memset(path, 0, sizeof(*path));
    path->ctx = ctx;
    path->window = window;
    path->window_bytes = ft_tt_tlb_window_size(window);

    if (path->window_bytes == 0u || (path->window_bytes & (path->window_bytes - 1u)) != 0u) {
        ft_tt_log_error("noc: window size %zu is not a power of two", path->window_bytes);
        memset(path, 0, sizeof(*path));
        return FT_TT_EIO;
    }
    return FT_TT_OK;
}

int ft_tt_noc_path_open(ft_tt_ctx *ctx, enum ft_tt_kmd_cache_mode cache, ft_tt_noc_path *path)
{
    ft_tt_tlb_window *window = NULL;
    int rc;

    if (!ctx || !path || !ft_tt_ctx_available(ctx))
        return FT_TT_EINVAL;

    memset(path, 0, sizeof(*path));

    /* 2 MiB is the only window size this board has — with no BAR4 the 4 GiB
     * kind has a count of zero and ALLOCATE_TLB rejects it (spec §2.3). */
    rc = ft_tt_tlb_window_alloc(ctx, (size_t)2u * 1024u * 1024u, cache, &window);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_noc_path_attach(ctx, window, path);
    if (rc != FT_TT_OK) {
        (void)ft_tt_tlb_window_free(ctx, window);
        memset(path, 0, sizeof(*path));
        return rc;
    }
    path->owns_window = true;
    return FT_TT_OK;
}

int ft_tt_noc_path_close(ft_tt_noc_path *path)
{
    int rc = FT_TT_OK;

    if (!path)
        return FT_TT_EINVAL;
    if (path->window && path->owns_window)
        rc = ft_tt_tlb_window_free(path->ctx, path->window);

    memset(path, 0, sizeof(*path));
    return rc;
}

uint64_t ft_tt_noc_path_aim_count(const ft_tt_noc_path *path)
{
    return path ? path->aim_count : 0u;
}

uint64_t ft_tt_noc_path_write_transactions(const ft_tt_noc_path *path)
{
    return path ? path->write_transactions : 0u;
}

uint64_t ft_tt_noc_path_read_transactions(const ft_tt_noc_path *path)
{
    return path ? path->read_transactions : 0u;
}

/* ------------------------------------------------------------------------- */
/* The window walk                                                           */
/* ------------------------------------------------------------------------- */

/**
 * Point the path's window at `base` on (x, y) unless it is already there.
 *
 * The skip is safe only because the path owns its window and is the sole thing
 * that aims it. It matters: a run of small transfers to one core would
 * otherwise pay 1.23 µs per call for a window that never moved.
 */
static int aim_if_needed(ft_tt_noc_path *path, uint32_t x, uint32_t y, uint64_t base)
{
    int rc;

    if (path->aimed && path->aimed_base == base && path->aimed_x == x && path->aimed_y == y)
        return FT_TT_OK;

    rc = ft_tt_tlb_window_aim(path->ctx, path->window, (uint16_t)x, (uint16_t)y, base);
    if (rc != FT_TT_OK) {
        path->aimed = false;
        return rc;
    }
    /* The window layer records the aim too; this is belt and braces against a
     * build where the Phase 1b fence swallowed it. */
    if (!ft_tt_tlb_window_is_aimed(path->window)) {
        path->aimed = false;
        return FT_TT_EPERM;
    }

    path->aimed = true;
    path->aimed_base = base;
    path->aimed_x = x;
    path->aimed_y = y;
    path->aim_count++;
    return FT_TT_OK;
}

int ft_tt_noc_window_slice_at(uint64_t address, size_t bytes, size_t window_bytes,
                              ft_tt_noc_window_slice *out_slice)
{
    if (ft_tt_noc_window_slice_compute(address, bytes, window_bytes, out_slice))
        return FT_TT_OK;

    if (out_slice)
        ft_tt_log_error("noc: window size %zu is not a power of two", window_bytes);
    return FT_TT_EINVAL;
}

int ft_tt_noc_transfer_raw(ft_tt_noc_path *path, ft_tt_noc_coord coord, uint64_t address,
                           void *host, size_t bytes, bool is_read, ft_tt_noc_transfer_width width)
{
    unsigned char *host_bytes = (unsigned char *)host;
    int rc;

    if (!path || !path->ctx || !path->window)
        return FT_TT_EINVAL;
    if (!host && bytes != 0u)
        return FT_TT_EINVAL;

    rc = ft_tt_noc_check_raw_transfer(width, address, bytes);
    if (rc != FT_TT_OK)
        return rc;

    /* x_end/y_end are 6-bit fields in the TLB register (docs/tt_driver_notes.md
     * §4). A coordinate that does not fit them would be silently truncated into
     * a different, live endpoint — refuse instead. */
    if (coord.x > 63u || coord.y > 63u) {
        ft_tt_log_error("noc: coordinate (%u, %u) does not fit the 6-bit TLB fields", coord.x,
                        coord.y);
        return FT_TT_EINVAL;
    }

    while (bytes > 0u) {
        ft_tt_noc_window_slice slice;
        volatile unsigned char *aperture;

        rc = ft_tt_noc_window_slice_at(address, bytes, path->window_bytes, &slice);
        if (rc != FT_TT_OK)
            return rc;

        rc = aim_if_needed(path, coord.x, coord.y, slice.window_base);
        if (rc != FT_TT_OK)
            return rc;

        aperture = (volatile unsigned char *)ft_tt_tlb_window_aperture(path->window);
        if (!aperture) {
            ft_tt_log_error("noc: window %u has no aperture", ft_tt_tlb_window_id(path->window));
            return FT_TT_EIO;
        }

        ft_tt_noc_copy_slice(aperture + slice.offset_in_window, host_bytes, slice.bytes, is_read,
                             width);

        address += slice.bytes;
        host_bytes += slice.bytes;
        bytes -= slice.bytes;

        /* The unit divides `slice.bytes` exactly: ft_tt_noc_check_raw_transfer() refused
         * any length that is not a multiple of it before a single byte moved. */
        {
            const uint64_t units = slice.bytes / ft_tt_noc_transfer_width_bytes(width);

            if (is_read) {
                path->bytes_read += slice.bytes;
                path->read_transactions += units;
            } else {
                path->bytes_written += slice.bytes;
                path->write_transactions += units;
            }
        }
    }
    return FT_TT_OK;
}

/**
 * Move `bytes` between `host` and (coord, address) with the memory space's own
 * rules applied first: this is ft_tt_noc_transfer_raw() plus
 * ft_tt_noc_check_transfer(), and the four named helpers are all one line of it.
 *
 * Nothing reaches the device until every argument has been checked: a refusal
 * here is free, and a bad endpoint is a power cycle.
 */
static int noc_transfer(ft_tt_noc_path *path, ft_tt_memory_space space, bool is_read,
                        ft_tt_noc_coord coord, uint64_t address, void *host, size_t bytes)
{
    int rc = ft_tt_noc_check_transfer(space, is_read, address, bytes);

    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_noc_transfer_raw(path, coord, address, host, bytes, is_read,
                                  FT_TT_NOC_WIDTH_BLOCK16);
}

/* ------------------------------------------------------------------------- */
/* Registers and mailbox fields — WORD32, no memory space, no bounds         */
/* ------------------------------------------------------------------------- */

int ft_tt_write_register_words(ft_tt_noc_path *path, ft_tt_noc_coord target, uint64_t address,
                               const void *src, size_t bytes)
{
    /* Const is dropped so one walk serves both directions; with is_read false
     * nothing is ever written through this pointer. */
    return ft_tt_noc_transfer_raw(path, target, address, (void *)(uintptr_t)src, bytes, false,
                                  FT_TT_NOC_WIDTH_WORD32);
}

int ft_tt_read_register_words(ft_tt_noc_path *path, ft_tt_noc_coord target, uint64_t address,
                              void *dst, size_t bytes)
{
    return ft_tt_noc_transfer_raw(path, target, address, dst, bytes, true, FT_TT_NOC_WIDTH_WORD32);
}

/* ------------------------------------------------------------------------- */
/* The four named helpers                                                    */
/* ------------------------------------------------------------------------- */

int ft_tt_write_l1(ft_tt_noc_path *path, ft_tt_noc_coord core, uint64_t address, const void *src,
                   size_t bytes)
{
    /* Const is dropped so one walk serves both directions; with is_read false
     * nothing is ever written through this pointer. */
    return noc_transfer(path, FT_TT_MEM_SPACE_L1, false, core, address, (void *)(uintptr_t)src,
                        bytes);
}

int ft_tt_read_l1(ft_tt_noc_path *path, ft_tt_noc_coord core, uint64_t address, void *dst,
                  size_t bytes)
{
    return noc_transfer(path, FT_TT_MEM_SPACE_L1, true, core, address, dst, bytes);
}

int ft_tt_write_dram(ft_tt_noc_path *path, ft_tt_noc_coord bank_port, uint64_t address,
                     const void *src, size_t bytes)
{
    return noc_transfer(path, FT_TT_MEM_SPACE_DRAM, false, bank_port, address,
                        (void *)(uintptr_t)src, bytes);
}

int ft_tt_read_dram(ft_tt_noc_path *path, ft_tt_noc_coord bank_port, uint64_t address, void *dst,
                    size_t bytes)
{
    return noc_transfer(path, FT_TT_MEM_SPACE_DRAM, true, bank_port, address, dst, bytes);
}
