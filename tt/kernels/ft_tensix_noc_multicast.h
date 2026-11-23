/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_noc_multicast.h
 * @brief NoC multicast of an L1 block to a rectangle of Tensix cores, plus the semaphore
 *        pieces of Foundry's sender/receiver rendezvous.
 *
 * Rendezvous (Foundry convention; nothing here enforces it):
 *   receiver: set its arrival semaphore INVALID, atomically increment the sender's ready
 *             semaphore, wait until arrival == VALID.
 *   sender:   wait until ready == receiver count, zero it, multicast the payload, wait for
 *             the writes-sent flush (the payload and the semaphore multicast go through
 *             different command buffers whose FIFOs are not ordered against each other), then
 *             multicast VALID into every receiver's arrival semaphore.
 * `destination_count` is the exact number of receivers, the sender excluded (a non-loopback
 * multicast never writes to its own core). The write barrier's acknowledgement accounting
 * depends on it being exact.
 *
 * Every issuing function checks its input at the point of issue and returns a code; a
 * refusal issues nothing and moves no counter. The return codes and semaphore values are
 * mirrored by value in host and protocol headers, so they must not change.
 *
 * @status Complete for multicast write, multicast semaphore set and the local waits.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_NOC_MULTICAST_H
#define FT_TENSIX_NOC_MULTICAST_H

#include <stdint.h>

#include "ft_tensix_dataflow.h"

/* ---- Return codes (Foundry convention) ---- */
#define FT_TENSIX_MULTICAST_OK 0
#define FT_TENSIX_MULTICAST_ERROR_ZERO_LENGTH (-1)
#define FT_TENSIX_MULTICAST_ERROR_ZERO_DESTINATIONS (-2)
#define FT_TENSIX_MULTICAST_ERROR_ALIGNMENT (-3)
#define FT_TENSIX_MULTICAST_ERROR_RECTANGLE (-4)
#define FT_TENSIX_MULTICAST_ERROR_WAIT_TIMEOUT (-5)
#define FT_TENSIX_MULTICAST_ERROR_STOPPED (-6)

/* ---- Semaphore values (Foundry convention) ---- */
#define FT_TENSIX_SEMAPHORE_INVALID 0u
#define FT_TENSIX_SEMAPHORE_VALID 1u

/** Packet-control word of a non-posted multicast write on the multicast VC. */
#define FT_TENSIX_MULTICAST_PACKET_CONTROL                                                     \
    (FT_BH_NOC_PACKET_COPY | FT_BH_NOC_PACKET_WRITE | FT_BH_NOC_PACKET_STATIC_VC_ENABLE |      \
     FT_BH_NOC_PACKET_STATIC_VC(FT_BH_NOC_MULTICAST_WRITE_VC) | FT_BH_NOC_PACKET_PATH_RESERVE | \
     FT_BH_NOC_PACKET_BROADCAST | FT_BH_NOC_PACKET_RESPONSE_MARKED)

/** Multicast address of `address` across the inclusive rectangle (x_low, y_low)-(x_high,
 *  y_high). NoC 1 broadcasts in the opposite direction, so its start corner is the high one.
 *  Writes *out_address only on success. Forced. */
static FT_BH_INLINE int ft_tensix_multicast_address(uint32_t x_low, uint32_t y_low,
                                                    uint32_t x_high, uint32_t y_high,
                                                    uint32_t address, uint64_t *out_address)
{
    if (x_low > x_high || y_low > y_high)
        return FT_TENSIX_MULTICAST_ERROR_RECTANGLE;
    if (x_high > FT_BH_NOC_NODE_ID_MASK || y_high > FT_BH_NOC_NODE_ID_MASK)
        return FT_TENSIX_MULTICAST_ERROR_RECTANGLE;
#if NOC_INDEX == 0
    *out_address = FT_BH_NOC_MULTICAST_ADDR(x_low, y_low, x_high, y_high, address);
#else
    *out_address = FT_BH_NOC_MULTICAST_ADDR(x_high, y_high, x_low, y_low, address);
#endif
    return FT_TENSIX_MULTICAST_OK;
}

/** Program and launch one multicast burst on the write buffer; the caller has waited for the
 *  buffer and guarantees 0 < bytes <= maximum burst. Forced. */
static FT_BH_INLINE void ft_tensix_noc_issue_multicast(uint32_t source_l1_address,
                                                       uint64_t destination, uint32_t bytes,
                                                       uint32_t destination_count)
{
    const uint32_t packet_control = FT_TENSIX_MULTICAST_PACKET_CONTROL;

    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_PACKET_CONTROL,
                        packet_control);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_TARGET_ADDR_LOW,
                        source_l1_address);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_RETURN_ADDR_LOW,
                        (uint32_t)destination);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_RETURN_ADDR_MID,
                        (uint32_t)(destination >> 32) & FT_BH_NOC_PCIE_HIGH_ADDR_MASK);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_RETURN_COORDINATE,
                        (uint32_t)(destination >> 36) & FT_BH_NOC_COORDINATE_MASK);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_ATOMIC_LEN_BE, bytes);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, FT_BH_NOC_REG_COMMAND_CONTROL,
                        FT_BH_NOC_COMMAND_SEND);
    noc_nonposted_writes_num_issued[NOC_INDEX] += 1;
    noc_nonposted_writes_acked[NOC_INDEX] += destination_count;
}

/** Multicast `bytes` of local L1 to every core of a multicast address, split into bursts.
 *  Length and both addresses must be 16-byte aligned. Plain. */
static inline int ft_tensix_noc_write_multicast(uint32_t source_l1_address, uint64_t destination,
                                                uint32_t bytes, uint32_t destination_count)
{
    uint32_t remaining = bytes;
    uint32_t source = source_l1_address;
    uint64_t target = destination;

    if (bytes == 0u)
        return FT_TENSIX_MULTICAST_ERROR_ZERO_LENGTH;
    if (destination_count == 0u)
        return FT_TENSIX_MULTICAST_ERROR_ZERO_DESTINATIONS;
    if ((bytes % FT_BH_NOC_L1_ALIGNMENT_BYTES) != 0u ||
        (source_l1_address % FT_BH_NOC_L1_ALIGNMENT_BYTES) != 0u ||
        (((uint32_t)destination) % FT_BH_NOC_L1_ALIGNMENT_BYTES) != 0u)
        return FT_TENSIX_MULTICAST_ERROR_ALIGNMENT;
    while (remaining > FT_BH_NOC_MAX_BURST_BYTES) {
        while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_WRITE_CMD_BUF))
            ;
        ft_tensix_noc_issue_multicast(source, target, FT_BH_NOC_MAX_BURST_BYTES,
                                      destination_count);
        source += FT_BH_NOC_MAX_BURST_BYTES;
        target += FT_BH_NOC_MAX_BURST_BYTES;
        remaining -= FT_BH_NOC_MAX_BURST_BYTES;
    }
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_WRITE_CMD_BUF))
        ;
    ft_tensix_noc_issue_multicast(source, target, remaining, destination_count);
    return FT_TENSIX_MULTICAST_OK;
}

/** Multicast the 4-byte word at local `source_l1_address` (a semaphore value) to every core
 *  of a multicast address, on the register-write command buffer. Plain. */
static inline int ft_tensix_noc_semaphore_set_multicast(uint32_t source_l1_address,
                                                        uint64_t destination,
                                                        uint32_t destination_count)
{
    const uint32_t packet_control = FT_TENSIX_MULTICAST_PACKET_CONTROL;

    if (destination_count == 0u)
        return FT_TENSIX_MULTICAST_ERROR_ZERO_DESTINATIONS;
    if ((source_l1_address % FT_BH_NOC_REG_ALIGNMENT_BYTES) != 0u ||
        (((uint32_t)destination) % FT_BH_NOC_REG_ALIGNMENT_BYTES) != 0u)
        return FT_TENSIX_MULTICAST_ERROR_ALIGNMENT;
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF))
        ;
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_PACKET_CONTROL,
                        packet_control);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_TARGET_ADDR_LOW,
                        source_l1_address);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_RETURN_ADDR_LOW,
                        (uint32_t)destination);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_RETURN_ADDR_MID,
                        (uint32_t)(destination >> 32) & FT_BH_NOC_PCIE_HIGH_ADDR_MASK);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_RETURN_COORDINATE,
                        (uint32_t)(destination >> 36) & FT_BH_NOC_COORDINATE_MASK);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_ATOMIC_LEN_BE,
                        FT_BH_NOC_REG_ALIGNMENT_BYTES);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_REG_WRITE_CMD_BUF, FT_BH_NOC_REG_COMMAND_CONTROL,
                        FT_BH_NOC_COMMAND_SEND);
    noc_nonposted_writes_num_issued[NOC_INDEX] += 1;
    noc_nonposted_writes_acked[NOC_INDEX] += destination_count;
    return FT_TENSIX_MULTICAST_OK;
}

/** Wait until *semaphore >= value. Bounded by `spin_limit` passes; returns early with
 *  STOPPED when `stop_flag` is non-null and set. Plain. */
static inline int ft_tensix_wait_semaphore_at_least(volatile uint32_t *semaphore, uint32_t value,
                                                    uint32_t spin_limit,
                                                    volatile const uint32_t *stop_flag)
{
    uint32_t spins = 0u;

    for (;;) {
        ft_bh_l1_cache_invalidate();
        if (*semaphore >= value)
            return FT_TENSIX_MULTICAST_OK;
        if (stop_flag != 0 && *stop_flag != 0u)
            return FT_TENSIX_MULTICAST_ERROR_STOPPED;
        if (++spins >= spin_limit)
            return FT_TENSIX_MULTICAST_ERROR_WAIT_TIMEOUT;
    }
}

/** Wait until *semaphore == value; bounds and stop as ft_tensix_wait_semaphore_at_least.
 *  Plain. */
static inline int ft_tensix_wait_semaphore_equals(volatile uint32_t *semaphore, uint32_t value,
                                                  uint32_t spin_limit,
                                                  volatile const uint32_t *stop_flag)
{
    uint32_t spins = 0u;

    for (;;) {
        ft_bh_l1_cache_invalidate();
        if (*semaphore == value)
            return FT_TENSIX_MULTICAST_OK;
        if (stop_flag != 0 && *stop_flag != 0u)
            return FT_TENSIX_MULTICAST_ERROR_STOPPED;
        if (++spins >= spin_limit)
            return FT_TENSIX_MULTICAST_ERROR_WAIT_TIMEOUT;
    }
}

/** Store `value` into a local semaphore word. Forced. */
static FT_BH_INLINE void ft_tensix_set_local_semaphore(volatile uint32_t *semaphore,
                                                       uint32_t value)
{
    *semaphore = value;
}

#endif /* FT_TENSIX_NOC_MULTICAST_H */
