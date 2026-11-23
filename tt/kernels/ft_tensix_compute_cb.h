/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_compute_cb.h
 * @brief Circular-buffer flow control for compute kernels: wait/pop on the UNPACK thread,
 *   reserve/push on the PACK thread, with the counter publication done by the coprocessor
 *   (SETDMAREG + STALLWAIT + STOREREG) so it lands after the unpacker/packer has finished.
 *
 * The CB bookkeeping struct and the two overlay stream registers per CB are owned by
 * ft_tensix_circular_buffer.h. Thread ownership is the kernel's gating, not checked here.
 *
 * @status Clean-room rewrite, plan Phase 4 step 4.5a.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_COMPUTE_CB_H
#define FT_TENSIX_COMPUTE_CB_H

#include <stdint.h>

#include "ft_tensix_circular_buffer.h"
#include "ft_tensix_compute.h"

/* STOREREG takes a register as an 18-bit word address (STOREREG.md): RISC byte address >> 2. */
#define FT_TENSIX_CB_TENSIX_REG_ADDRESS(byte_address) ((((uint32_t)(byte_address)) >> 2) & 0x3ffffu)

/* GPR the unpack side stages the acked count in (Foundry convention). */
#define FT_TENSIX_CB_GPR_UNPACK_ACKED 4u

/** Block until `tile_count` tiles are available in CB `cb_index`. [UNPACK] */
static FT_BH_INLINE void ft_tensix_compute_cb_wait_front(uint32_t cb_index, uint32_t tile_count)
{
    volatile FT_BH_REG_PTR uint32_t *p = ft_tensix_cb_pages_received_pointer((int32_t)cb_index);
    uint16_t wanted = (uint16_t)tile_count;
    uint16_t received;
    uint16_t available;
    do {
        received = (uint16_t)p[0];
        available = (uint16_t)(received - ft_tensix_cb(cb_index)->tiles.counters.tiles_acked);
    } while (available < wanted);
}

/** Release `tile_count` tiles: publish the acked count, then advance the read pointer.
 *  [UNPACK] */
static FT_BH_INLINE void ft_tensix_compute_cb_pop_front(uint32_t cb_index, uint32_t tile_count)
{
    ft_tensix_cb_interface *cb = ft_tensix_cb(cb_index);
    uint32_t reg = FT_TENSIX_CB_TENSIX_REG_ADDRESS(
        ft_tensix_cb_pages_acked_pointer((int32_t)cb_index));
    uint32_t byte_count = tile_count * cb->fifo_page_size;

    cb->tiles.counters.tiles_acked = (uint16_t)(cb->tiles.counters.tiles_acked + tile_count);
    FT_BH_PUSH_SETDMAREG(0, cb->tiles.counters.tiles_acked, 0,
                         FT_TENSIX_LO_16(FT_TENSIX_CB_GPR_UNPACK_ACKED));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_THCON, FT_BH_STALLWAIT_COND_UNPACK);
    FT_BH_PUSH_STOREREG(FT_TENSIX_CB_GPR_UNPACK_ACKED, reg);

    cb->fifo_read_pointer += byte_count;
    if (cb->fifo_read_pointer >= cb->fifo_limit) {
        cb->fifo_read_pointer -= cb->fifo_size;
    }
}

/** Block until `tile_count` tiles of space are free. Polls the acked register against this
 *  thread's own received count. [PACK] */
static FT_BH_INLINE void ft_tensix_compute_cb_reserve_back(uint32_t cb_index, uint32_t tile_count)
{
    volatile FT_BH_REG_PTR uint32_t *p = ft_tensix_cb_pages_acked_pointer((int32_t)cb_index);
    uint16_t received = ft_tensix_cb(cb_index)->tiles.counters.tiles_received;
    int32_t free_tiles;
    do {
        uint16_t acked = (uint16_t)p[0];
        uint16_t free_wrapped = (uint16_t)(ft_tensix_cb(cb_index)->fifo_num_pages
                                           - (uint16_t)(received - acked));
        free_tiles = (int32_t)free_wrapped;
    } while (free_tiles < (int32_t)tile_count);
}

/** Commit `tile_count` tiles: advance the write pointer, then publish the received count.
 *  [PACK] */
static FT_BH_INLINE void ft_tensix_compute_cb_push_back(uint32_t cb_index, uint32_t tile_count)
{
    ft_tensix_cb_interface *cb = ft_tensix_cb(cb_index);
    uint32_t byte_count = tile_count * cb->fifo_page_size;
    uint32_t reg = FT_TENSIX_CB_TENSIX_REG_ADDRESS(
        ft_tensix_cb_pages_received_pointer((int32_t)cb_index));

    cb->fifo_write_pointer += byte_count;
    cb->fifo_write_tile_pointer = 0;
    if (cb->fifo_write_pointer >= cb->fifo_limit) {
        cb->fifo_write_pointer -= cb->fifo_size;
    }
    cb->tiles.counters.tiles_received =
        (uint16_t)(cb->tiles.counters.tiles_received + tile_count);
    FT_BH_PUSH_SETDMAREG(0, cb->tiles.counters.tiles_received, 0,
                         FT_TENSIX_LO_16(FT_TENSIX_GPR_PACK_NUM_MSGS_RECEIVED));
    FT_BH_TTINSN_STALLWAIT(FT_BH_STALLWAIT_BLOCK_THCON, FT_BH_STALLWAIT_COND_PACK);
    FT_BH_PUSH_STOREREG(FT_TENSIX_GPR_PACK_NUM_MSGS_RECEIVED, reg);
}

#endif /* FT_TENSIX_COMPUTE_CB_H */
