/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_arc_msg.c
 * @brief Implementation of the ARC firmware message queue encoder and
 *   protocol (see ft_tt_udrv_arc_msg.h).
 *
 * @status Off-hardware only — no transport here issues a NoC transaction.
 * @issues None
 * @todo None
 */

#include "ft_tt_udrv_arc_msg.h"

#include <string.h>

#include "ft_tt_udrv.h"

/** Words a message occupies on the wire. */
#define ARC_MSG_WORDS (1u + FT_TT_UDRV_ARC_PAYLOAD_WORDS)

/** True when `[address, address + bytes)` lies inside the ARC's CSM. */
static bool is_within_csm(uint64_t address, uint64_t bytes)
{
    uint64_t limit = (uint64_t)FT_TT_UDRV_ARC_CSM_BASE + (uint64_t)FT_TT_UDRV_ARC_CSM_SIZE;

    if (bytes == 0u)
        return false;

    if (address < (uint64_t)FT_TT_UDRV_ARC_CSM_BASE)
        return false;

    return (address + bytes) <= limit;
}

int ft_tt_udrv_arc_queue_init(uint32_t qcb_word0, uint32_t qcb_word1,
                              ft_tt_udrv_arc_queue *out_queue)
{
    uint64_t ring_bytes;
    uint32_t num_entries = qcb_word1 & 0xFFu;

    if (!out_queue)
        return FT_TT_UDRV_EINVAL;

    /* A zero-entry queue divides by zero in every slot computation below. The
     * kernel driver carries the same guard. */
    if (num_entries == 0u)
        return FT_TT_UDRV_EINVAL;

    ring_bytes = (uint64_t)FT_TT_UDRV_ARC_QUEUE_HEADER_BYTES +
                 (2u * (uint64_t)num_entries * (uint64_t)FT_TT_UDRV_ARC_MSG_BYTES);

    if (!is_within_csm(qcb_word0, ring_bytes))
        return FT_TT_UDRV_EINVAL;

    memset(out_queue, 0, sizeof(*out_queue));
    out_queue->base = qcb_word0;
    out_queue->num_entries = num_entries;
    out_queue->request_base = qcb_word0 + FT_TT_UDRV_ARC_QUEUE_HEADER_BYTES;
    out_queue->response_base =
        out_queue->request_base + (num_entries * FT_TT_UDRV_ARC_MSG_BYTES);
    out_queue->req_wptr_addr = qcb_word0 + 0x00u;
    out_queue->res_rptr_addr = qcb_word0 + 0x04u;
    out_queue->req_rptr_addr = qcb_word0 + 0x10u;
    out_queue->res_wptr_addr = qcb_word0 + 0x14u;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_ring_occupancy(uint32_t wptr, uint32_t rptr, uint32_t num_entries,
                                  uint32_t *out_occupancy)
{
    uint32_t modulus;

    if (!out_occupancy || num_entries == 0u)
        return FT_TT_UDRV_EINVAL;

    modulus = 2u * num_entries;

    /* A pointer outside [0, 2N) cannot be told apart from a full ring. Refusing is the difference between a diagnosed queue and a
     * message written to an address the firmware never reads. */
    if (wptr >= modulus || rptr >= modulus)
        return FT_TT_UDRV_EINVAL;

    *out_occupancy = (wptr >= rptr) ? (wptr - rptr) : (modulus - (rptr - wptr));
    return FT_TT_UDRV_OK;
}

/** Shared slot-address arithmetic for the two rings. */
static int slot_word_address(uint32_t ring_base, uint32_t pointer, uint32_t num_entries,
                             uint32_t word_index, uint32_t *out_address)
{
    uint32_t slot;

    if (!out_address || num_entries == 0u || word_index >= ARC_MSG_WORDS)
        return FT_TT_UDRV_EINVAL;

    if (pointer >= (2u * num_entries))
        return FT_TT_UDRV_EINVAL;

    slot = pointer % num_entries;
    *out_address = ring_base + (slot * FT_TT_UDRV_ARC_MSG_BYTES) + (word_index * 4u);
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_request_word_address(const ft_tt_udrv_arc_queue *queue, uint32_t wptr,
                                        uint32_t word_index, uint32_t *out_address)
{
    if (!queue)
        return FT_TT_UDRV_EINVAL;

    return slot_word_address(queue->request_base, wptr, queue->num_entries, word_index,
                             out_address);
}

int ft_tt_udrv_arc_response_word_address(const ft_tt_udrv_arc_queue *queue, uint32_t rptr,
                                         uint32_t word_index, uint32_t *out_address)
{
    if (!queue)
        return FT_TT_UDRV_EINVAL;

    return slot_word_address(queue->response_base, rptr, queue->num_entries, word_index,
                             out_address);
}

int ft_tt_udrv_arc_ring_advance(uint32_t pointer, uint32_t num_entries, uint32_t *out_pointer)
{
    uint32_t modulus;

    if (!out_pointer || num_entries == 0u)
        return FT_TT_UDRV_EINVAL;

    modulus = 2u * num_entries;
    if (pointer >= modulus)
        return FT_TT_UDRV_EINVAL;

    *out_pointer = (pointer + 1u) % modulus;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_msg_word(const ft_tt_udrv_arc_msg *msg, uint32_t word_index,
                            uint32_t *out_value)
{
    if (!msg || !out_value || word_index >= ARC_MSG_WORDS)
        return FT_TT_UDRV_EINVAL;

    *out_value = (word_index == 0u) ? msg->header : msg->payload[word_index - 1u];
    return FT_TT_UDRV_OK;
}

/* ------------------------------------------------------------------------ *
 * Builders
 * ------------------------------------------------------------------------ */

/** Zero the message and set its header. */
static int build_header_only(ft_tt_udrv_arc_msg *out_msg, uint32_t header)
{
    if (!out_msg)
        return FT_TT_UDRV_EINVAL;

    memset(out_msg, 0, sizeof(*out_msg));
    out_msg->header = header;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_msg_asic_state0(ft_tt_udrv_arc_msg *out_msg)
{
    return build_header_only(out_msg, FT_TT_UDRV_ARC_MSG_ASIC_STATE0);
}

int ft_tt_udrv_arc_msg_asic_state3(ft_tt_udrv_arc_msg *out_msg)
{
    return build_header_only(out_msg, FT_TT_UDRV_ARC_MSG_ASIC_STATE3);
}

int ft_tt_udrv_arc_msg_test(ft_tt_udrv_arc_msg *out_msg)
{
    return build_header_only(out_msg, FT_TT_UDRV_ARC_MSG_TEST);
}

int ft_tt_udrv_arc_msg_set_wdt_timeout(uint32_t milliseconds, ft_tt_udrv_arc_msg *out_msg)
{
    int rc = build_header_only(out_msg, FT_TT_UDRV_ARC_MSG_SET_WDT_TIMEOUT);

    if (rc != FT_TT_UDRV_OK)
        return rc;

    out_msg->payload[0] = milliseconds;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_msg_trigger_reset(uint16_t argument, ft_tt_udrv_arc_msg *out_msg)
{
    int rc = build_header_only(out_msg, FT_TT_UDRV_ARC_MSG_TRIGGER_RESET);

    if (rc != FT_TT_UDRV_OK)
        return rc;

    out_msg->payload[0] = argument;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_msg_power_setting(uint8_t validity, uint16_t power_flags,
                                     const uint16_t *settings, ft_tt_udrv_arc_msg *out_msg)
{
    uint32_t word;
    int rc;

    rc = build_header_only(out_msg, FT_TT_UDRV_ARC_MSG_POWER_SETTING |
                                        ((uint32_t)validity << 8) |
                                        ((uint32_t)power_flags << 16));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (!settings)
        return FT_TT_UDRV_OK;

    /* Fourteen u16 settings into seven u32 payload words. The driver does this
     * with memcpy, which is a host-endianness assumption; spelling it out
     * makes the WIRE format the definition instead. */
    for (word = 0u; word < FT_TT_UDRV_ARC_PAYLOAD_WORDS; word++)
        out_msg->payload[word] =
            (uint32_t)settings[word * 2u] | ((uint32_t)settings[(word * 2u) + 1u] << 16);

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_msg_power_setting_open_default(ft_tt_udrv_arc_msg *out_msg)
{
    return ft_tt_udrv_arc_msg_power_setting(
        0x0Fu, (uint16_t)(FT_TT_UDRV_ARC_POWER_FLAG_ALL & ~FT_TT_UDRV_ARC_POWER_FLAG_MAX_AI_CLK),
        NULL, out_msg);
}

int ft_tt_udrv_arc_msg_power_setting_last_close(ft_tt_udrv_arc_msg *out_msg)
{
    return ft_tt_udrv_arc_msg_power_setting(0x0Fu, 0u, NULL, out_msg);
}

/* ------------------------------------------------------------------------ *
 * Protocol
 * ------------------------------------------------------------------------ */

/** True when the transport can perform every operation the protocol uses. */
static bool transport_is_usable(const ft_tt_udrv_arc_transport *transport)
{
    return transport && transport->csm_read32 && transport->csm_write32 && transport->trigger;
}

/**
 * Read a ring pointer, treating all-ones as a dead device.
 *
 * 0xFFFFFFFF is what a NoC read returns when the fabric is hung or the BARs
 * have gone away, and tt-kmd checks for it on every pointer read. Without the check a hung card reads as a ring with
 * absurd pointers and the failure surfaces as arithmetic.
 */
static int read_pointer(const ft_tt_udrv_arc_transport *transport, uint32_t address,
                        uint32_t *out_value)
{
    if (transport->csm_read32(transport->context, address, out_value) != 0)
        return FT_TT_UDRV_EIO;

    if (*out_value == 0xFFFFFFFFu)
        return FT_TT_UDRV_EIO;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_try_push(const ft_tt_udrv_arc_transport *transport,
                            const ft_tt_udrv_arc_queue *queue, const ft_tt_udrv_arc_msg *msg)
{
    uint32_t wptr = 0;
    uint32_t rptr = 0;
    uint32_t occupancy = 0;
    uint32_t word;
    int rc;

    if (!transport_is_usable(transport) || !queue || !msg)
        return FT_TT_UDRV_EINVAL;

    rc = read_pointer(transport, queue->req_wptr_addr, &wptr);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = read_pointer(transport, queue->req_rptr_addr, &rptr);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = ft_tt_udrv_arc_ring_occupancy(wptr, rptr, queue->num_entries, &occupancy);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (occupancy >= queue->num_entries)
        return FT_TT_UDRV_EAGAIN;

    for (word = 0u; word < ARC_MSG_WORDS; word++) {
        uint32_t address = 0;
        uint32_t value = 0;

        rc = ft_tt_udrv_arc_request_word_address(queue, wptr, word, &address);
        if (rc != FT_TT_UDRV_OK)
            return rc;

        rc = ft_tt_udrv_arc_msg_word(msg, word, &value);
        if (rc != FT_TT_UDRV_OK)
            return rc;

        if (transport->csm_write32(transport->context, address, value) != 0)
            return FT_TT_UDRV_EIO;
    }

    rc = ft_tt_udrv_arc_ring_advance(wptr, queue->num_entries, &wptr);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    /* The pointer bump is what publishes the message, so it must be the LAST
     * write and it must follow every payload word. */
    if (transport->csm_write32(transport->context, queue->req_wptr_addr, wptr) != 0)
        return FT_TT_UDRV_EIO;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_try_pop(const ft_tt_udrv_arc_transport *transport,
                           const ft_tt_udrv_arc_queue *queue, ft_tt_udrv_arc_msg *out_msg)
{
    uint32_t wptr = 0;
    uint32_t rptr = 0;
    uint32_t occupancy = 0;
    uint32_t word;
    int rc;

    if (!transport_is_usable(transport) || !queue || !out_msg)
        return FT_TT_UDRV_EINVAL;

    rc = read_pointer(transport, queue->res_rptr_addr, &rptr);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = read_pointer(transport, queue->res_wptr_addr, &wptr);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = ft_tt_udrv_arc_ring_occupancy(wptr, rptr, queue->num_entries, &occupancy);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (occupancy == 0u)
        return FT_TT_UDRV_EAGAIN;

    memset(out_msg, 0, sizeof(*out_msg));

    for (word = 0u; word < ARC_MSG_WORDS; word++) {
        uint32_t address = 0;
        uint32_t value = 0;

        rc = ft_tt_udrv_arc_response_word_address(queue, rptr, word, &address);
        if (rc != FT_TT_UDRV_OK)
            return rc;

        if (transport->csm_read32(transport->context, address, &value) != 0)
            return FT_TT_UDRV_EIO;

        if (word == 0u)
            out_msg->header = value;
        else
            out_msg->payload[word - 1u] = value;
    }

    rc = ft_tt_udrv_arc_ring_advance(rptr, queue->num_entries, &rptr);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (transport->csm_write32(transport->context, queue->res_rptr_addr, rptr) != 0)
        return FT_TT_UDRV_EIO;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_arc_send_sync(const ft_tt_udrv_arc_transport *transport,
                             const ft_tt_udrv_arc_queue *queue, ft_tt_udrv_arc_msg *msg,
                             uint32_t max_polls)
{
    ft_tt_udrv_arc_msg drained;
    uint32_t attempt;
    uint32_t poll;
    int rc;

    if (!transport_is_usable(transport) || !queue || !msg)
        return FT_TT_UDRV_EINVAL;

    /* Discard anything stale a previous exchange left behind, bounded by the
     * ring size so a runaway pointer cannot spin forever. Nothing legally
     * produces more than num_entries responses. */
    for (attempt = 0u; attempt < queue->num_entries; attempt++)
        if (ft_tt_udrv_arc_try_pop(transport, queue, &drained) != FT_TT_UDRV_OK)
            break;

    rc = ft_tt_udrv_arc_try_push(transport, queue, msg);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (transport->trigger(transport->context) != 0)
        return FT_TT_UDRV_EIO;

    for (poll = 0u;; poll++) {
        rc = ft_tt_udrv_arc_try_pop(transport, queue, msg);
        if (rc != FT_TT_UDRV_EAGAIN)
            break;

        if (poll >= max_polls)
            return FT_TT_UDRV_EAGAIN;

        if (transport->wait)
            transport->wait(transport->context);
    }

    if (rc != FT_TT_UDRV_OK)
        return rc;

    /* A response header of zero is success; anything else is the firmware's
     * own status and the caller needs to see it, so it is left in place. */
    if (msg->header != 0u)
        return FT_TT_UDRV_EIO;

    return FT_TT_UDRV_OK;
}
