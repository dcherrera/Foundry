/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_noc.c
 * @brief Implementation of ft_noc.h: BRISC firmware's one-time NIU setup.
 *
 * Plain 32-bit stores to the NIU's own registers. Nothing here issues a NoC transaction, and
 * in particular nothing uses an inline write (the recorded boot-wedge hazard).
 *
 * @status Clean-room replacement of the staged tree's noc.c, plan Phase 6.
 * @issues None
 * @todo None
 */
#include "ft_noc.h"

#define FT_NOC_INITIATOR_1_CTRL 0x2090u /* measured preset (spec §7.4) */
#define FT_NOC_COORDINATE_MASK  0xFFFu

static inline volatile uint32_t *niu_register(uint32_t noc, uint32_t offset)
{
    return (volatile uint32_t *)(FT_NOC_NIU_BASE(noc) + offset);
}

void ft_noc_set_niu_and_router_config(void)
{
    for (uint32_t noc = 0; noc < FT_NOC_COUNT; noc++) {
        *niu_register(noc, FT_NOC_NIU_CFG_0) |= 1u;
        *niu_register(noc, FT_NOC_ROUTER_CFG_0) |= 1u;
    }
}

void ft_noc_preset_command_buffers(uint32_t noc, uint32_t atomic_return_address)
{
    uint32_t coordinate = *niu_register(noc, FT_NOC_ID_LOGICAL) & FT_NOC_COORDINATE_MASK;

    *niu_register(noc, FT_NOC_INITIATOR(0) + FT_NOC_TARG_ADDR_MID) = 0u;
    *niu_register(noc, FT_NOC_INITIATOR(0) + FT_NOC_TARG_ADDR_HI) = coordinate;
    *niu_register(noc, FT_NOC_INITIATOR(2) + FT_NOC_TARG_ADDR_MID) = 0u;
    *niu_register(noc, FT_NOC_INITIATOR(2) + FT_NOC_TARG_ADDR_HI) = coordinate;
    *niu_register(noc, FT_NOC_INITIATOR(3) + FT_NOC_RET_ADDR_LO) = atomic_return_address;
    *niu_register(noc, FT_NOC_INITIATOR(3) + FT_NOC_RET_ADDR_MID) = 0u;
    *niu_register(noc, FT_NOC_INITIATOR(3) + FT_NOC_RET_ADDR_HI) = coordinate;
    *niu_register(noc, FT_NOC_INITIATOR(1) + FT_NOC_CTRL) = FT_NOC_INITIATOR_1_CTRL;
    *niu_register(noc, FT_NOC_INITIATOR(1) + FT_NOC_RET_ADDR_MID) = 0u;
    *niu_register(noc, FT_NOC_INITIATOR(1) + FT_NOC_RET_ADDR_HI) = coordinate;
}

void ft_noc_clear_packet_tags(uint32_t noc)
{
    for (uint32_t initiator = 0; initiator < 4u; initiator++)
        *niu_register(noc, FT_NOC_INITIATOR(initiator) + FT_NOC_PACKET_TAG) = 0u;
}

void ft_noc_seed_software_counters(uint32_t noc, const ft_noc_software_counters *counters)
{
    counters->reads_issued[noc] = *niu_register(noc, FT_NOC_RD_RESP_RECEIVED);
    counters->nonposted_writes_issued[noc] = *niu_register(noc, FT_NOC_NONPOSTED_WR_REQ_SENT);
    counters->nonposted_writes_acked[noc] = *niu_register(noc, FT_NOC_WR_ACK_RECEIVED);
    counters->nonposted_atomics_acked[noc] = *niu_register(noc, FT_NOC_ATOMIC_RESP_RECEIVED);
    counters->posted_writes_issued[noc] = *niu_register(noc, FT_NOC_POSTED_WR_REQ_SENT);
}
