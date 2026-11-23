/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_transfer_guard.h
 * @brief Predicates a resident data-movement kernel checks before it issues a NoC transfer
 *        or dereferences a pointer that arrived in a descriptor.
 *
 * The host-side checker never sees transfers a kernel issues on its own, so these are the
 * primary defence: non-zero length, inside L1, above the firmware's read-only region, both
 * ends congruent modulo the NoC alignment, overflow-free products, and the CB block-grid
 * invariant. Each returns `false` instead of faulting; the caller turns that into an error
 * code and drains. FT_TENSIX_CB_ASSERT is only the backstop behind them.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_TRANSFER_GUARD_H
#define FT_TENSIX_TRANSFER_GUARD_H

#include <stdbool.h>
#include <stdint.h>

#include "../fw/ft_bh_regs.h"
#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_noc.h"

/** Lowest L1 byte a NoC write or wire pointer may target; below it are the firmware's
 *  mailboxes, core info, go message and fabric block (equals the host's
 *  FT_TT_MEM_MAP_READ_ONLY_END; tests/test_memory_ops.c pins them equal). */
#define FT_TENSIX_L1_WRITABLE_BASE 37120u

/** NoC alignment of L1 (and DRAM write) transfers, in bytes. */
#define FT_TENSIX_L1_ALIGNMENT 16u

static FT_BH_INLINE bool ft_tensix_transfer_length_is_issuable(uint32_t length_bytes)
{
    return length_bytes != 0u;
}

/** Non-empty span that does not wrap and ends inside L1. */
static FT_BH_INLINE bool ft_tensix_local_l1_span_is_readable(uint32_t address,
                                                             uint32_t length_bytes)
{
    if (!ft_tensix_transfer_length_is_issuable(length_bytes))
        return false;
    if (address + length_bytes < address)
        return false;
    return address + length_bytes <= FT_TENSIX_L1_SIZE_BYTES;
}

/** Readable span that also starts above the firmware's read-only region. */
static FT_BH_INLINE bool ft_tensix_local_l1_span_is_writable(uint32_t address,
                                                             uint32_t length_bytes)
{
    if (!ft_tensix_local_l1_span_is_readable(address, length_bytes))
        return false;
    return address >= FT_TENSIX_L1_WRITABLE_BASE;
}

/** Word-aligned writable span (for a block the kernel stores through a pointer). */
static FT_BH_INLINE bool ft_tensix_l1_block_is_writable(uint32_t address, uint32_t length_bytes)
{
    if ((address & 0x3u) != 0u)
        return false;
    return ft_tensix_local_l1_span_is_writable(address, length_bytes);
}

/** Both ends have the same offset modulo `alignment` (a power of two). */
static FT_BH_INLINE bool ft_tensix_transfer_ends_are_congruent(uint32_t local_address,
                                                               uint32_t remote_address,
                                                               uint32_t alignment)
{
    const uint32_t mask = alignment - 1u;

    return (local_address & mask) == (remote_address & mask);
}

/** An L1-to-L1 copy: readable source, writable destination, congruent ends. */
static FT_BH_INLINE bool ft_tensix_l1_copy_is_issuable(uint32_t source_address,
                                                       uint32_t destination_address,
                                                       uint32_t length_bytes)
{
    if (!ft_tensix_local_l1_span_is_readable(source_address, length_bytes))
        return false;
    if (!ft_tensix_local_l1_span_is_writable(destination_address, length_bytes))
        return false;
    return ft_tensix_transfer_ends_are_congruent(source_address, destination_address,
                                                 FT_TENSIX_L1_ALIGNMENT);
}

/** left * right without 32-bit overflow; *out_product is written only on success. */
static FT_BH_INLINE bool ft_tensix_product_fits(uint32_t left, uint32_t right,
                                                uint32_t *out_product)
{
    if (left != 0u && right > 0xFFFFFFFFu / left)
        return false;
    *out_product = left * right;
    return true;
}

/** A reservation of `chunk_pages` is satisfiable, blocks of that size tile the FIFO exactly,
 *  and `pointer` sits on that block grid now (pointers wrap only on an exact limit hit). */
static FT_BH_INLINE bool ft_tensix_cb_block_grid_is_safe(const ft_tensix_cb_interface *interface,
                                                         uint32_t chunk_pages, uint32_t pointer)
{
    uint32_t base;
    uint32_t block_bytes;

    if (interface->fifo_num_pages == 0u || interface->fifo_page_size == 0u || chunk_pages == 0u)
        return false;
    if (chunk_pages > interface->fifo_num_pages ||
        (interface->fifo_num_pages % chunk_pages) != 0u)
        return false;
    if (chunk_pages > 0xFFFFFFFFu / interface->fifo_page_size)
        return false;
    block_bytes = chunk_pages * interface->fifo_page_size;
    if (interface->fifo_limit < interface->fifo_size)
        return false;
    base = interface->fifo_limit - interface->fifo_size;
    if (pointer < base || pointer >= interface->fifo_limit)
        return false;
    return ((pointer - base) % block_bytes) == 0u;
}

/** Block-grid check of CB `cb_index`'s write pointer. */
static FT_BH_INLINE bool ft_tensix_cb_push_grid_is_safe(uint32_t cb_index, uint32_t chunk_pages)
{
    const ft_tensix_cb_interface *interface = ft_tensix_cb(cb_index);

    return ft_tensix_cb_block_grid_is_safe(interface, chunk_pages, interface->fifo_write_pointer);
}

/** Block-grid check of CB `cb_index`'s read pointer. */
static FT_BH_INLINE bool ft_tensix_cb_pop_grid_is_safe(uint32_t cb_index, uint32_t chunk_pages)
{
    const ft_tensix_cb_interface *interface = ft_tensix_cb(cb_index);

    return ft_tensix_cb_block_grid_is_safe(interface, chunk_pages, interface->fifo_read_pointer);
}

#endif /* FT_TENSIX_TRANSFER_GUARD_H */
