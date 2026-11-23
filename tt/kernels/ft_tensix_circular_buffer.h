/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_circular_buffer.h
 * @brief Circular-buffer (CB) records shared with the firmware, and data-movement CB flow
 *        control: wait for pages, wait for space, publish pages, free pages.
 *
 * Each core has 64 CBs. CB n is paired with overlay stream n, whose two 16-bit wrapping
 * registers count pages RECEIVED (bumped by the producer) and pages ACKED (bumped by the
 * consumer); their difference is the number of readable pages. The per-CB record below holds
 * the FIFO geometry and both pointers; the firmware fills it at launch and kernels advance it.
 * Pointers wrap only when they land exactly on the limit, by subtracting the size.
 *
 * The record layout is a binary contract with the firmware (32 bytes). Its units are bytes on
 * the data-movement cores and 16-byte words on the compute cores; this header does not care.
 * It is compiled into data-movement kernels, compute kernels and firmware alike, so it uses
 * nothing that only exists on one processor.
 *
 * @status Complete for the data-movement side.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_CIRCULAR_BUFFER_H
#define FT_TENSIX_CIRCULAR_BUFFER_H

#include <stdint.h>

#include "../fw/ft_bh_regs.h"

/** CBs (and overlay streams) per Blackhole Tensix core. */
#ifndef FT_TENSIX_NUM_CIRCULAR_BUFFERS
#define FT_TENSIX_NUM_CIRCULAR_BUFFERS 64
#endif

/** Backstop check on CB pointer arithmetic: traps with `ebreak` when the build enables
 *  lightweight kernel asserts, otherwise only type-checks the condition. */
#if defined(LIGHTWEIGHT_KERNEL_ASSERTS)
#define FT_TENSIX_CB_ASSERT(condition) \
    do {                               \
        if (!(condition)) {            \
            __asm__("ebreak");         \
        }                              \
    } while (0)
#else
#define FT_TENSIX_CB_ASSERT(condition) ((void)sizeof((condition) ? 1 : 0))
#endif

/** One CB's bookkeeping record (firmware contract, 32 bytes). `fifo_limit` is base + size. */
typedef struct {
    uint32_t fifo_size;
    uint32_t fifo_limit;
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;
    uint32_t fifo_read_pointer;
    uint32_t fifo_write_pointer;
    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        } counters;
    } tiles;
    uint32_t fifo_write_tile_pointer;
} ft_tensix_cb_interface;

_Static_assert(sizeof(ft_tensix_cb_interface) == 32, "CB record is a 32-byte firmware contract");

/** The firmware's CB records, resolved into kernels by the firmware-symbols link. */
extern ft_tensix_cb_interface cb_interface[FT_TENSIX_NUM_CIRCULAR_BUFFERS];

static FT_BH_INLINE ft_tensix_cb_interface *ft_tensix_cb(uint32_t cb_index)
{
    return &cb_interface[cb_index];
}

/** Stream register counting pages received into CB `cb_index`. */
static FT_BH_INLINE volatile FT_BH_REG_PTR uint32_t *
ft_tensix_cb_pages_received_pointer(int32_t cb_index)
{
    return (volatile FT_BH_REG_PTR uint32_t *)(uintptr_t)FT_BH_STREAM_REG_ADDR(
        cb_index, FT_BH_STREAM_PAGES_RECEIVED_INDEX);
}

/** Stream register counting pages acknowledged (freed) from CB `cb_index`. */
static FT_BH_INLINE volatile FT_BH_REG_PTR uint32_t *
ft_tensix_cb_pages_acked_pointer(int32_t cb_index)
{
    return (volatile FT_BH_REG_PTR uint32_t *)(uintptr_t)FT_BH_STREAM_REG_ADDR(
        cb_index, FT_BH_STREAM_PAGES_ACKED_INDEX);
}

/** Consumer: block until `page_count` pages are readable at the front. */
static FT_BH_INLINE void ft_tensix_cb_wait_front(int32_t cb_index, int32_t page_count)
{
    uint32_t pages_acked = ft_tensix_cb_pages_acked_pointer(cb_index)[0];
    uintptr_t pages_received_pointer = (uintptr_t)ft_tensix_cb_pages_received_pointer(cb_index);
    uint16_t pages_received;

    do {
        pages_received = ((uint16_t)ft_bh_reg_read(pages_received_pointer)) - pages_acked;
    } while (pages_received < page_count);
}

/** Producer: block until `page_count` pages are free at the back. */
static FT_BH_INLINE void ft_tensix_cb_reserve_back(int32_t cb_index, int32_t page_count)
{
    uintptr_t pages_acked_pointer = (uintptr_t)ft_tensix_cb_pages_acked_pointer(cb_index);
    uint32_t pages_received = ft_tensix_cb_pages_received_pointer(cb_index)[0];
    int32_t free_space_pages;

    do {
        ft_bh_l1_cache_invalidate();
        uint16_t pages_acked = (uint16_t)ft_bh_reg_read(pages_acked_pointer);
        uint16_t free_space_pages_wrap =
            ft_tensix_cb(cb_index)->fifo_num_pages - ((uint16_t)pages_received - pages_acked);
        free_space_pages = (int32_t)free_space_pages_wrap;
    } while (free_space_pages < page_count);
}

/** Producer: publish `page_count` pages and advance the write pointer. */
static FT_BH_INLINE void ft_tensix_cb_push_back(int32_t cb_index, int32_t page_count)
{
    uint32_t num_words = page_count * ft_tensix_cb(cb_index)->fifo_page_size;
    volatile FT_BH_REG_PTR uint32_t *pages_received = ft_tensix_cb_pages_received_pointer(cb_index);

    pages_received[0] += page_count;
    ft_tensix_cb(cb_index)->fifo_write_pointer += num_words;
    FT_TENSIX_CB_ASSERT(ft_tensix_cb(cb_index)->fifo_write_pointer <=
                        ft_tensix_cb(cb_index)->fifo_limit);
    if (ft_tensix_cb(cb_index)->fifo_write_pointer == ft_tensix_cb(cb_index)->fifo_limit)
        ft_tensix_cb(cb_index)->fifo_write_pointer -= ft_tensix_cb(cb_index)->fifo_size;
}

/** Consumer: free `page_count` pages and advance the read pointer. */
static FT_BH_INLINE void ft_tensix_cb_pop_front(int32_t cb_index, int32_t page_count)
{
    volatile FT_BH_REG_PTR uint32_t *pages_acked = ft_tensix_cb_pages_acked_pointer(cb_index);

    pages_acked[0] += page_count;
    uint32_t num_words = page_count * ft_tensix_cb(cb_index)->fifo_page_size;
    ft_tensix_cb(cb_index)->fifo_read_pointer += num_words;
    FT_TENSIX_CB_ASSERT(ft_tensix_cb(cb_index)->fifo_read_pointer <=
                        ft_tensix_cb(cb_index)->fifo_limit);
    if (ft_tensix_cb(cb_index)->fifo_read_pointer == ft_tensix_cb(cb_index)->fifo_limit)
        ft_tensix_cb(cb_index)->fifo_read_pointer -= ft_tensix_cb(cb_index)->fifo_size;
}

/** First address of CB `cb_index`'s FIFO. */
static FT_BH_INLINE uint32_t ft_tensix_cb_base_pointer(uint32_t cb_index)
{
    return ft_tensix_cb(cb_index)->fifo_limit - ft_tensix_cb(cb_index)->fifo_size;
}

/** Put the write pointer back on the FIFO base. The caller holds the whole buffer reserved;
 *  resident kernels do this at descriptor boundaries because an op's traffic need not be a
 *  whole number of buffers. */
static FT_BH_INLINE void ft_tensix_cb_rebase_write_pointer(uint32_t cb_index)
{
    ft_tensix_cb(cb_index)->fifo_write_pointer = ft_tensix_cb_base_pointer(cb_index);
    ft_tensix_cb(cb_index)->fifo_write_tile_pointer = 0u;
}

/** Put the read pointer back on the FIFO base (see ft_tensix_cb_rebase_write_pointer). */
static FT_BH_INLINE void ft_tensix_cb_rebase_read_pointer(uint32_t cb_index)
{
    ft_tensix_cb(cb_index)->fifo_read_pointer = ft_tensix_cb_base_pointer(cb_index);
}

static FT_BH_INLINE uint32_t ft_tensix_cb_write_pointer(uint32_t cb_index)
{
    return ft_tensix_cb(cb_index)->fifo_write_pointer;
}

static FT_BH_INLINE uint32_t ft_tensix_cb_read_pointer(uint32_t cb_index)
{
    return ft_tensix_cb(cb_index)->fifo_read_pointer;
}

#endif /* FT_TENSIX_CIRCULAR_BUFFER_H */
