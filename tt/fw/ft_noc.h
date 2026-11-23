/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_noc.h
 * @brief NoC interface unit (NIU) setup that BRISC firmware performs once at boot, for both
 *   NoCs: the NIU/router configuration bits, the static command-buffer presets the kernels'
 *   NoC code relies on, and the software transaction counters.
 *
 * Registers are tt-isa-documentation BlackholeA0/NoC/MemoryMap.md and Counters.md. The
 * values written are the measured contract of the firmware this replaces (spec §7.4).
 *
 * @status Clean-room replacement of the staged tree's noc.c, plan Phase 6.
 * @issues None
 * @todo None
 */
#ifndef FT_NOC_H
#define FT_NOC_H

#include <stdint.h>

#define FT_NOC_COUNT 2u

/* MemoryMap.md: NIU_BASE per NoC, four request initiators NOC_CMD_BUF_OFFSET apart. */
#define FT_NOC_NIU_BASE(noc)        (0xFFB20000u + (uint32_t)(noc) * 0x10000u)
#define FT_NOC_INITIATOR(index)     ((uint32_t)(index) * 0x800u)
#define FT_NOC_TARG_ADDR_MID        0x04u
#define FT_NOC_TARG_ADDR_HI         0x08u
#define FT_NOC_RET_ADDR_LO          0x0Cu
#define FT_NOC_RET_ADDR_MID         0x10u
#define FT_NOC_RET_ADDR_HI          0x14u
#define FT_NOC_PACKET_TAG           0x18u
#define FT_NOC_CTRL                 0x1Cu
#define FT_NOC_NIU_CFG_0            0x100u
#define FT_NOC_ROUTER_CFG_0         0x104u
#define FT_NOC_ID_LOGICAL           0x148u /* low 12 bits: x | y << 6 */

/* Counters.md: the NIU counter array at NIU_BASE + 0x200. */
#define FT_NOC_COUNTER(index)           (0x200u + 4u * (uint32_t)(index))
#define FT_NOC_ATOMIC_RESP_RECEIVED     FT_NOC_COUNTER(0)
#define FT_NOC_WR_ACK_RECEIVED          FT_NOC_COUNTER(1)
#define FT_NOC_RD_RESP_RECEIVED         FT_NOC_COUNTER(2)
#define FT_NOC_NONPOSTED_WR_REQ_SENT    FT_NOC_COUNTER(10)
#define FT_NOC_POSTED_WR_REQ_SENT       FT_NOC_COUNTER(11)

/** The software counters kernels keep in step with the NIU's (fw/ft_dataflow_entry.c). */
typedef struct {
    uint32_t *reads_issued;
    uint32_t *nonposted_writes_issued;
    uint32_t *nonposted_writes_acked;
    uint32_t *nonposted_atomics_acked;
    uint32_t *posted_writes_issued;
} ft_noc_software_counters;

/** Set bit 0 of NIU_CFG_0 and ROUTER_CFG_0 on both NoCs (documented as reserved; the measured
 *  boot sets it, spec §7.4). */
void ft_noc_set_niu_and_router_config(void);

/** The static command-buffer presets on one NoC: this tile's coordinate as the target of
 *  initiators 0 and 2 and the return of initiators 1 and 3, initiator 1's control word, and
 *  initiator 3's atomic return address. Kernels rewrite only some fields per transaction. */
void ft_noc_preset_command_buffers(uint32_t noc, uint32_t atomic_return_address);

/** Zero NOC_PACKET_TAG of all four initiators on one NoC. */
void ft_noc_clear_packet_tags(uint32_t noc);

/** Copy the NIU's transaction counters for `noc` into the software counters. */
void ft_noc_seed_software_counters(uint32_t noc, const ft_noc_software_counters *counters);

#endif /* FT_NOC_H */
