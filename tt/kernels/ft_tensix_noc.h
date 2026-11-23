/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_noc.h
 * @brief Unicast NoC transfers, a remote atomic increment and the NIU-counter barriers for a
 *        Blackhole data-movement core (BRISC or NCRISC).
 *
 * A transfer is programmed into one of the NIU's command buffers (target, return, length,
 * packet control) and launched by a store to the command-control register. Completion is
 * observed by comparing the NIU's status counters with software counters that every issue
 * path here bumps. Those software counters belong to the firmware: it defines them, and it
 * checks them against the hardware after the kernel returns, so each issue path must move
 * exactly the counters it documents.
 *
 * Linkage: functions marked "forced" are always inlined. The six plain `static inline`
 * functions (read, write, semaphore increment, read barrier, writes-flushed barrier, inline
 * word write) may be kept out of line by the compiler; their definition order is kept fixed
 * because it can decide the order of those copies in .text.
 *
 * Register layout: BlackholeA0/NoC/MemoryMap.md and the NIU counter list in the same tree.
 *
 * @status Complete for the transfers Foundry kernels issue.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_NOC_H
#define FT_TENSIX_NOC_H

#include <stdint.h>

#include "../fw/ft_bh_regs.h"

/* ---- Software transfer counters (defined by the firmware image, one per NoC) ---- */

extern uint32_t noc_reads_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
extern uint32_t noc_nonposted_writes_acked[NUM_NOCS];
extern uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
extern uint32_t noc_posted_writes_num_issued[NUM_NOCS];

/** Size of one Tensix core's L1 (1.5 MiB). */
#define FT_TENSIX_L1_SIZE_BYTES (1536u * 1024u)

/** Result of ft_tensix_noc_inline_dword_write_l1 (Foundry convention). */
enum {
    FT_TENSIX_NOC_ISSUED = 0,
    FT_TENSIX_NOC_REFUSED_ALIGNMENT = 1,
    FT_TENSIX_NOC_REFUSED_NOT_L1 = 2
};

/** Packet-tag register of a command buffer; the transaction id is 4 bits at bit 10. */
#define FT_TENSIX_NOC_PACKET_TAG (FT_BH_NOC_REG_BASE + 0x18u)
#define FT_TENSIX_NOC_PACKET_TAG_TRID_SHIFT 10u
#define FT_TENSIX_NOC_PACKET_TAG_TRID_MASK 0xFu

/** NIU status counter "requests outstanding with transaction id `id`" (one per id 0..15). */
#define FT_TENSIX_NIU_MST_REQS_OUTSTANDING_ID(id) (0x10u + (id))

/** 64-bit NoC address of `address` in the core at (noc_x, noc_y). Forced. */
static FT_BH_INLINE uint64_t ft_tensix_noc_address(uint32_t noc_x, uint32_t noc_y,
                                                   uint32_t address)
{
    return FT_BH_NOC_XY_ADDR(noc_x, noc_y, address);
}

/** Program and launch one read burst; the caller has waited for the buffer. Forced.
 *  The packet-control register is left alone: the firmware's setting stands for reads. */
static FT_BH_INLINE void ft_tensix_noc_issue_read(uint32_t noc, uint32_t command_buffer,
                                                  uint64_t source_address,
                                                  uint32_t destination_address,
                                                  uint32_t length_bytes,
                                                  uint32_t read_request_vc)
{
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_RETURN_ADDR_LOW, destination_address);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_TARGET_ADDR_LOW,
                        (uint32_t)source_address);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_TARGET_ADDR_MID,
                        (uint32_t)(source_address >> 32) & FT_BH_NOC_PCIE_HIGH_ADDR_MASK);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_TARGET_COORDINATE,
                        (uint32_t)(source_address >> FT_BH_NOC_COORDINATE_SHIFT) &
                            FT_BH_NOC_COORDINATE_MASK);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_ATOMIC_LEN_BE, length_bytes);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_COMMAND_CONTROL,
                        FT_BH_NOC_COMMAND_SEND);
    noc_reads_num_issued[noc] += 1;
    (void)read_request_vc;
}

/** Program and launch one non-posted unicast write burst (one acknowledgement). Forced. */
static FT_BH_INLINE void ft_tensix_noc_issue_write(uint32_t noc, uint32_t command_buffer,
                                                   uint32_t source_address,
                                                   uint64_t destination_address,
                                                   uint32_t length_bytes, uint32_t vc)
{
    uint32_t packet_control = FT_BH_NOC_PACKET_COPY | FT_BH_NOC_PACKET_WRITE |
                              FT_BH_NOC_PACKET_STATIC_VC_ENABLE | FT_BH_NOC_PACKET_STATIC_VC(vc) |
                              FT_BH_NOC_PACKET_RESPONSE_MARKED;

    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_PACKET_CONTROL, packet_control);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_TARGET_ADDR_LOW, source_address);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_RETURN_ADDR_LOW,
                        (uint32_t)destination_address);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_RETURN_ADDR_MID,
                        (uint32_t)(destination_address >> 32) & FT_BH_NOC_PCIE_HIGH_ADDR_MASK);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_RETURN_COORDINATE,
                        (uint32_t)(destination_address >> 36) & FT_BH_NOC_COORDINATE_MASK);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_ATOMIC_LEN_BE, length_bytes);
    ft_bh_noc_cmd_write(noc, command_buffer, FT_BH_NOC_REG_COMMAND_CONTROL,
                        FT_BH_NOC_COMMAND_SEND);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
}

/* ---- Counter predicates: hardware counter equals the software one. Forced. ---- */

static FT_BH_INLINE int ft_tensix_noc_reads_flushed(uint32_t noc)
{
    return ft_bh_noc_status_read(noc, FT_BH_NOC_COUNTER_READ_RESPONSES_RECEIVED) ==
           noc_reads_num_issued[noc];
}

static FT_BH_INLINE int ft_tensix_noc_writes_flushed(uint32_t noc)
{
    return ft_bh_noc_status_read(noc, FT_BH_NOC_COUNTER_WRITE_ACKS_RECEIVED) ==
           noc_nonposted_writes_acked[noc];
}

static FT_BH_INLINE int ft_tensix_noc_writes_sent(uint32_t noc)
{
    return ft_bh_noc_status_read(noc, FT_BH_NOC_COUNTER_NONPOSTED_WRITES_SENT) ==
           noc_nonposted_writes_num_issued[noc];
}

static FT_BH_INLINE int ft_tensix_noc_atomics_flushed(uint32_t noc)
{
    return ft_bh_noc_status_read(noc, FT_BH_NOC_COUNTER_ATOMIC_RESPONSES_RECEIVED) ==
           noc_nonposted_atomics_acked[noc];
}

static FT_BH_INLINE int ft_tensix_noc_posted_writes_sent(uint32_t noc)
{
    return ft_bh_noc_status_read(noc, FT_BH_NOC_COUNTER_POSTED_WRITES_SENT) ==
           noc_posted_writes_num_issued[noc];
}

/** Read `length_bytes` (non-zero) from NoC address `source` into local L1 `destination`,
 *  split into maximum-size bursts. Plain. */
static inline void ft_tensix_noc_read(uint64_t source, uint32_t destination,
                                      uint32_t length_bytes)
{
    while (length_bytes > FT_BH_NOC_MAX_BURST_BYTES) {
        while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_READ_CMD_BUF))
            ;
        ft_tensix_noc_issue_read(NOC_INDEX, FT_BH_READ_CMD_BUF, source, destination,
                                 FT_BH_NOC_MAX_BURST_BYTES, 1);
        source += FT_BH_NOC_MAX_BURST_BYTES;
        destination += FT_BH_NOC_MAX_BURST_BYTES;
        length_bytes -= FT_BH_NOC_MAX_BURST_BYTES;
    }
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_READ_CMD_BUF))
        ;
    ft_tensix_noc_issue_read(NOC_INDEX, FT_BH_READ_CMD_BUF, source, destination, length_bytes, 1);
}

/** Write `length_bytes` (non-zero) of local L1 `source` to NoC address `destination`,
 *  split into maximum-size non-posted bursts. Plain. */
static inline void ft_tensix_noc_write(uint32_t source, uint64_t destination,
                                       uint32_t length_bytes)
{
    while (length_bytes > FT_BH_NOC_MAX_BURST_BYTES) {
        while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_WRITE_CMD_BUF))
            ;
        ft_tensix_noc_issue_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, source, destination,
                                  FT_BH_NOC_MAX_BURST_BYTES, FT_BH_NOC_UNICAST_WRITE_VC);
        source += FT_BH_NOC_MAX_BURST_BYTES;
        destination += FT_BH_NOC_MAX_BURST_BYTES;
        length_bytes -= FT_BH_NOC_MAX_BURST_BYTES;
    }
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_WRITE_CMD_BUF))
        ;
    ft_tensix_noc_issue_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, source, destination, length_bytes,
                              FT_BH_NOC_UNICAST_WRITE_VC);
}

/** Non-posted remote atomic: add `increment` to the 32-bit word at `destination`. Plain. */
static inline void ft_tensix_noc_semaphore_increment(uint64_t destination, uint32_t increment)
{
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF))
        ;
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_TARGET_ADDR_LOW,
                        (uint32_t)destination);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_TARGET_ADDR_MID,
                        (uint32_t)(destination >> 32) & FT_BH_NOC_PCIE_HIGH_ADDR_MASK);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_TARGET_COORDINATE,
                        (uint32_t)(destination >> 36) & FT_BH_NOC_COORDINATE_MASK);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_PACKET_CONTROL,
                        FT_BH_NOC_PACKET_STATIC_VC_ENABLE |
                            FT_BH_NOC_PACKET_STATIC_VC(FT_BH_NOC_UNICAST_WRITE_VC) |
                            FT_BH_NOC_PACKET_RESPONSE_MARKED | FT_BH_NOC_PACKET_ATOMIC);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_ATOMIC_LEN_BE,
                        FT_BH_NOC_ATOMIC_OPCODE(FT_BH_NOC_ATOMIC_OP_INCREMENT_GET) |
                            FT_BH_NOC_ATOMIC_WRAP(31u) |
                            FT_BH_NOC_ATOMIC_WORD_INDEX((((uint32_t)destination) >> 2) & 0x3u) |
                            FT_BH_NOC_ATOMIC_SOURCE_INDEX(0u));
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_ATOMIC_DATA, increment);
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_ATOMIC_CMD_BUF, FT_BH_NOC_REG_COMMAND_CONTROL,
                        FT_BH_NOC_COMMAND_SEND);
    noc_nonposted_atomics_acked[NOC_INDEX] += 1;
}

/** Wait until every issued read has landed, then drop stale cached lines. Plain. */
static inline void ft_tensix_noc_read_barrier(void)
{
    while (!ft_tensix_noc_reads_flushed(NOC_INDEX))
        ;
    ft_bh_l1_cache_invalidate();
}

/** Wait until every non-posted write is acknowledged. Forced. */
static FT_BH_INLINE void ft_tensix_noc_write_barrier(void)
{
    while (!ft_tensix_noc_writes_flushed(NOC_INDEX))
        ;
    ft_bh_l1_cache_invalidate();
}

/** Wait until every non-posted write has left this core (sources reusable). Plain. */
static inline void ft_tensix_noc_writes_flushed_barrier(void)
{
    while (!ft_tensix_noc_writes_sent(NOC_INDEX))
        ;
}

/** Wait until every issued atomic has its response. Forced. */
static FT_BH_INLINE void ft_tensix_noc_atomic_barrier(void)
{
    while (!ft_tensix_noc_atomics_flushed(NOC_INDEX))
        ;
}

/** Write one 32-bit word into another core's L1 by staging it in this processor's own slot
 *  (Foundry L1 layout: one 16-byte slot per processor and NoC, word at the destination's
 *  offset within 16 bytes) and issuing a 4-byte non-posted write. Refusals issue nothing.
 *  Plain. */
static inline int ft_tensix_noc_inline_dword_write_l1(uint64_t destination, uint32_t value)
{
    const uint32_t local = (uint32_t)destination;
    uint32_t staging;
    volatile FT_BH_L1_PTR uint32_t *staging_word;

    if ((local & 0x3u) != 0u)
        return FT_TENSIX_NOC_REFUSED_ALIGNMENT;
    if (local >= FT_TENSIX_L1_SIZE_BYTES)
        return FT_TENSIX_NOC_REFUSED_NOT_L1;
    staging = FT_BH_L1_INLINE_STAGING_BASE +
              (2u * FT_BH_L1_INLINE_STAGING_PER_NOC) * FT_BH_PROCESSOR_TYPE +
              NOC_INDEX * FT_BH_L1_INLINE_STAGING_PER_NOC + (local & 0xFu);
    while (!ft_tensix_noc_writes_sent(NOC_INDEX))
        ;
    while (!ft_tensix_noc_posted_writes_sent(NOC_INDEX))
        ;
    staging_word = (volatile FT_BH_L1_PTR uint32_t *)(uintptr_t)staging;
    staging_word[0] = value;
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_WRITE_CMD_BUF))
        ;
    ft_tensix_noc_issue_write(NOC_INDEX, FT_BH_WRITE_CMD_BUF, staging, destination, 4u,
                              FT_BH_NOC_UNICAST_WRITE_VC);
    return FT_TENSIX_NOC_ISSUED;
}

/** Tag later reads on the read command buffer with transaction id `trid` (sticky). Forced. */
static FT_BH_INLINE void ft_tensix_noc_read_set_trid(uint32_t trid)
{
    while (!ft_bh_noc_cmd_buf_ready(NOC_INDEX, FT_BH_READ_CMD_BUF))
        ;
    ft_bh_noc_cmd_write(NOC_INDEX, FT_BH_READ_CMD_BUF, FT_TENSIX_NOC_PACKET_TAG,
                        (trid & FT_TENSIX_NOC_PACKET_TAG_TRID_MASK)
                            << FT_TENSIX_NOC_PACKET_TAG_TRID_SHIFT);
}

/** Wait until no read tagged `trid` is outstanding, then drop stale cached lines. Forced. */
static FT_BH_INLINE void ft_tensix_noc_read_barrier_trid(uint32_t trid)
{
    while (ft_bh_noc_status_read(NOC_INDEX, FT_TENSIX_NIU_MST_REQS_OUTSTANDING_ID(trid)) != 0u)
        ;
    ft_bh_l1_cache_invalidate();
}

#endif /* FT_TENSIX_NOC_H */
