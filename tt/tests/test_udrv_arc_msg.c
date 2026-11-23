/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_arc_msg.c
 * @brief Gate for ft_tt_udrv_arc_msg.c — message encoding, ring arithmetic,
 *   and the synchronous exchange, driven against a SIMULATED ARC.
 *
 *   The fake firmware in this file implements the PEER side of the protocol
 *   from the driver's semantics — it maintains its own pointers with its own
 *   arithmetic and never calls a function under test. That is deliberate: a
 *   ring exerciser built out of the ring code proves the code is
 *   self-consistent and nothing else (DEV_NOTES §1, "never instrument a
 *   mechanism with itself").
 *
 *   WHAT THIS CANNOT PROVE: that a CSM write lands, that the doorbell is at
 *   0x800B0000, that the QCB pointer is in RESET_SCRATCH(11), or that real
 *   firmware answers any of these messages. Those need silicon. What it does
 *   prove is that every word this driver would put on the wire is the word the
 *   kernel would have put there, and that the ring bookkeeping survives wrap,
 *   fullness, staleness and a dead device.
 *
 * @status Off-hardware. No transport here performs I/O of any kind.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_arc_msg.h"

/** Where the fake queue lives inside the fake CSM. */
#define FAKE_QUEUE_BASE (FT_TT_UDRV_ARC_CSM_BASE + 0x1000u)
#define FAKE_ENTRIES 4u

/** Enough CSM to hold the header plus both rings, with room to spare. */
#define FAKE_CSM_WORDS 4096u

/**
 * A simulated ARC.
 *
 * `memory` is word-addressed from FT_TT_UDRV_ARC_CSM_BASE. `respond` decides
 * whether the doorbell produces a reply, and `status` is the header the reply
 * carries — zero for success, which is the firmware's convention.
 */
typedef struct fake_arc {
    uint32_t memory[FAKE_CSM_WORDS];
    bool respond;
    uint32_t status;
    bool dead;           /**< every read returns 0xFFFFFFFF */
    uint32_t reads;
    uint32_t writes;
    uint32_t triggers;
    uint32_t waits;
    uint32_t responses_pushed;
    ft_tt_udrv_arc_msg last_request; /**< what the firmware saw */
} fake_arc;

/** Word index of a CSM address, or FAKE_CSM_WORDS when out of range. */
static uint32_t fake_word_index(uint32_t address)
{
    uint32_t offset;

    if (address < FT_TT_UDRV_ARC_CSM_BASE || (address & 3u) != 0u)
        return FAKE_CSM_WORDS;

    offset = (address - FT_TT_UDRV_ARC_CSM_BASE) / 4u;
    return (offset < FAKE_CSM_WORDS) ? offset : FAKE_CSM_WORDS;
}

static int fake_read32(void *context, uint32_t address, uint32_t *out_value)
{
    fake_arc *arc = context;
    uint32_t index = fake_word_index(address);

    arc->reads++;

    if (arc->dead) {
        *out_value = 0xFFFFFFFFu;
        return 0;
    }

    if (index >= FAKE_CSM_WORDS)
        return -1;

    *out_value = arc->memory[index];
    return 0;
}

static int fake_write32(void *context, uint32_t address, uint32_t value)
{
    fake_arc *arc = context;
    uint32_t index = fake_word_index(address);

    arc->writes++;

    if (index >= FAKE_CSM_WORDS)
        return -1;

    arc->memory[index] = value;
    return 0;
}

/**
 * The doorbell: consume one request and produce one response.
 *
 * Written from the protocol description, not from the code under test — its
 * pointers, slot arithmetic and modulus are all its own.
 */
static int fake_trigger(void *context)
{
    fake_arc *arc = context;
    uint32_t header_word = (FAKE_QUEUE_BASE - FT_TT_UDRV_ARC_CSM_BASE) / 4u;
    uint32_t req_wptr = arc->memory[header_word + 0u];
    uint32_t res_rptr = arc->memory[header_word + 1u];
    uint32_t req_rptr = arc->memory[header_word + 4u];
    uint32_t res_wptr = arc->memory[header_word + 5u];
    uint32_t request_word_base;
    uint32_t response_word_base;
    uint32_t slot;
    uint32_t i;

    arc->triggers++;

    if (!arc->respond)
        return 0;

    /* Nothing to do if the request ring is empty. */
    if (req_wptr == req_rptr)
        return 0;

    request_word_base = header_word + (FT_TT_UDRV_ARC_QUEUE_HEADER_BYTES / 4u);
    response_word_base = request_word_base + (FAKE_ENTRIES * (FT_TT_UDRV_ARC_MSG_BYTES / 4u));

    slot = req_rptr % FAKE_ENTRIES;
    arc->last_request.header = arc->memory[request_word_base + (slot * 8u)];
    for (i = 0u; i < FT_TT_UDRV_ARC_PAYLOAD_WORDS; i++)
        arc->last_request.payload[i] = arc->memory[request_word_base + (slot * 8u) + 1u + i];

    req_rptr = (req_rptr + 1u) % (2u * FAKE_ENTRIES);
    arc->memory[header_word + 4u] = req_rptr;

    /* Echo the payload back with the firmware's status in the header word,
     * which is what makes "the response is the request's answer" checkable. */
    slot = res_wptr % FAKE_ENTRIES;
    arc->memory[response_word_base + (slot * 8u)] = arc->status;
    for (i = 0u; i < FT_TT_UDRV_ARC_PAYLOAD_WORDS; i++)
        arc->memory[response_word_base + (slot * 8u) + 1u + i] =
            arc->last_request.payload[i] ^ 0xA5A5A5A5u;

    res_wptr = (res_wptr + 1u) % (2u * FAKE_ENTRIES);
    arc->memory[header_word + 5u] = res_wptr;
    arc->responses_pushed++;

    (void)res_rptr;
    return 0;
}

static void fake_wait(void *context)
{
    fake_arc *arc = context;
    arc->waits++;
}

static void fake_init(fake_arc *arc, ft_tt_udrv_arc_transport *transport)
{
    memset(arc, 0, sizeof(*arc));
    arc->respond = true;
    arc->status = 0u;

    transport->csm_read32 = fake_read32;
    transport->csm_write32 = fake_write32;
    transport->trigger = fake_trigger;
    transport->wait = fake_wait;
    transport->context = arc;
}

/* ------------------------------------------------------------------------ *
 * Stages
 * ------------------------------------------------------------------------ */

static void stage_queue_location(void)
{
    ft_tt_udrv_arc_queue queue;

    ft_tt_test_step("Stage 1 — locating the queue, and the addresses that follow");

    ft_tt_test_check_fmt("QCB pointer is RESET_SCRATCH(11) = 0x8003042C",
                         FT_TT_UDRV_ARC_MSG_QCB_PTR == 0x8003042Cu, "0x%08x",
                         FT_TT_UDRV_ARC_MSG_QCB_PTR);
    ft_tt_test_check_fmt("boot status is RESET_SCRATCH(2) = 0x80030408",
                         FT_TT_UDRV_ARC_BOOT_STATUS == 0x80030408u, "0x%08x",
                         FT_TT_UDRV_ARC_BOOT_STATUS);

    ft_tt_test_check("a queue with 4 entries at CSM+0x1000 is accepted",
                     ft_tt_udrv_arc_queue_init(FAKE_QUEUE_BASE, FAKE_ENTRIES, &queue) ==
                         FT_TT_UDRV_OK,
                     "");
    ft_tt_test_check_fmt("entry count is the low byte of QCB word 1",
                         queue.num_entries == FAKE_ENTRIES, "%u", queue.num_entries);
    ft_tt_test_check_fmt("requests start 32 bytes past the base",
                         queue.request_base == FAKE_QUEUE_BASE + 32u, "0x%08x",
                         queue.request_base);
    ft_tt_test_check_fmt("responses start after all 4 requests",
                         queue.response_base == FAKE_QUEUE_BASE + 32u + (4u * 32u), "0x%08x",
                         queue.response_base);
    ft_tt_test_check_fmt("the four pointer addresses are +0, +4, +0x10, +0x14",
                         queue.req_wptr_addr == FAKE_QUEUE_BASE + 0x00u &&
                             queue.res_rptr_addr == FAKE_QUEUE_BASE + 0x04u &&
                             queue.req_rptr_addr == FAKE_QUEUE_BASE + 0x10u &&
                             queue.res_wptr_addr == FAKE_QUEUE_BASE + 0x14u,
                         "%08x %08x %08x %08x", queue.req_wptr_addr, queue.res_rptr_addr,
                         queue.req_rptr_addr, queue.res_wptr_addr);

    /* The low byte only: firmware puts other things in the upper bits. */
    ft_tt_test_check_fmt("only the low byte of QCB word 1 is the entry count",
                         ft_tt_udrv_arc_queue_init(FAKE_QUEUE_BASE, 0xDEAD0004u, &queue) ==
                                 FT_TT_UDRV_OK &&
                             queue.num_entries == 4u,
                         "%u", queue.num_entries);

    ft_tt_test_check("a zero-entry queue is refused",
                     ft_tt_udrv_arc_queue_init(FAKE_QUEUE_BASE, 0u, &queue) == FT_TT_UDRV_EINVAL,
                     "would divide by zero");

    ft_tt_test_check("a queue below the CSM is refused",
                     ft_tt_udrv_arc_queue_init(0x1000u, 4u, &queue) == FT_TT_UDRV_EINVAL,
                     "outside 0x10000000");

    ft_tt_test_check("a queue running off the end of the CSM is refused",
                     ft_tt_udrv_arc_queue_init(FT_TT_UDRV_ARC_CSM_BASE +
                                                   FT_TT_UDRV_ARC_CSM_SIZE - 64u,
                                               4u, &queue) == FT_TT_UDRV_EINVAL,
                     "the ring would not fit");

    ft_tt_test_check("a 255-entry queue that does not fit is refused",
                     ft_tt_udrv_arc_queue_init(FT_TT_UDRV_ARC_CSM_BASE +
                                                   FT_TT_UDRV_ARC_CSM_SIZE - 0x2000u,
                                               255u, &queue) == FT_TT_UDRV_EINVAL,
                     "255 * 2 * 32 = 16320 bytes");
}

static void stage_ring_arithmetic(void)
{
    uint32_t occupancy = 0u;
    uint32_t pointer = 0u;

    ft_tt_test_step("Stage 2 — ring arithmetic, including a non-power-of-two ring");

    ft_tt_test_check_fmt("empty ring", ft_tt_udrv_arc_ring_occupancy(0u, 0u, 4u, &occupancy) ==
                                              FT_TT_UDRV_OK &&
                                          occupancy == 0u,
                         "%u", occupancy);
    ft_tt_test_check_fmt("three in flight",
                         ft_tt_udrv_arc_ring_occupancy(3u, 0u, 4u, &occupancy) == FT_TT_UDRV_OK &&
                             occupancy == 3u,
                         "%u", occupancy);
    ft_tt_test_check_fmt("full ring (wptr has lapped into the second half)",
                         ft_tt_udrv_arc_ring_occupancy(4u, 0u, 4u, &occupancy) == FT_TT_UDRV_OK &&
                             occupancy == 4u,
                         "%u", occupancy);
    ft_tt_test_check_fmt("wrapped: wptr 1, rptr 7, 4 entries",
                         ft_tt_udrv_arc_ring_occupancy(1u, 7u, 4u, &occupancy) == FT_TT_UDRV_OK &&
                             occupancy == 2u,
                         "%u", occupancy);

    ft_tt_test_check("a pointer at or past 2N is refused",
                     ft_tt_udrv_arc_ring_occupancy(8u, 0u, 4u, &occupancy) == FT_TT_UDRV_EINVAL,
                     "a wild pointer must not read as a full ring");

    /* N not a power of two: a single u32 modular subtraction would get this wrong. */
    occupancy = 0xFFFFFFFFu;
    ft_tt_test_check_fmt("wrapped with N=3 (wptr 1, rptr 5): occupancy 2",
                         ft_tt_udrv_arc_ring_occupancy(1u, 5u, 3u, &occupancy) == FT_TT_UDRV_OK &&
                             occupancy == 2u,
                         "%u", occupancy);

    ft_tt_test_check_fmt("advance wraps at 2N",
                         ft_tt_udrv_arc_ring_advance(7u, 4u, &pointer) == FT_TT_UDRV_OK &&
                             pointer == 0u,
                         "%u", pointer);
    ft_tt_test_check_fmt("advance is ordinary in the middle",
                         ft_tt_udrv_arc_ring_advance(3u, 4u, &pointer) == FT_TT_UDRV_OK &&
                             pointer == 4u,
                         "%u", pointer);
}

static void stage_slot_addresses(void)
{
    ft_tt_udrv_arc_queue queue;
    uint32_t address = 0u;

    ft_tt_test_step("Stage 3 — slot addresses fold the pointer modulo N, not 2N");

    (void)ft_tt_udrv_arc_queue_init(FAKE_QUEUE_BASE, FAKE_ENTRIES, &queue);

    ft_tt_test_check_fmt("request slot 0 word 0",
                         ft_tt_udrv_arc_request_word_address(&queue, 0u, 0u, &address) ==
                                 FT_TT_UDRV_OK &&
                             address == queue.request_base,
                         "0x%08x", address);

    ft_tt_test_check_fmt("request slot 0 word 7 is 28 bytes in",
                         ft_tt_udrv_arc_request_word_address(&queue, 0u, 7u, &address) ==
                                 FT_TT_UDRV_OK &&
                             address == queue.request_base + 28u,
                         "0x%08x", address);

    /* wptr 4 and wptr 0 are the same SLOT — the extra bit only distinguishes
     * "full" from "empty", which is the whole reason the pointers run to 2N. */
    ft_tt_test_check_fmt("wptr 4 addresses the same slot as wptr 0",
                         ft_tt_udrv_arc_request_word_address(&queue, 4u, 0u, &address) ==
                                 FT_TT_UDRV_OK &&
                             address == queue.request_base,
                         "0x%08x", address);

    ft_tt_test_check_fmt("response slot 3 word 0",
                         ft_tt_udrv_arc_response_word_address(&queue, 3u, 0u, &address) ==
                                 FT_TT_UDRV_OK &&
                             address == queue.response_base + (3u * 32u),
                         "0x%08x", address);

    ft_tt_test_check("word 8 is refused (a message is 8 words)",
                     ft_tt_udrv_arc_request_word_address(&queue, 0u, 8u, &address) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check("a pointer at 2N is refused",
                     ft_tt_udrv_arc_request_word_address(&queue, 8u, 0u, &address) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check_fmt("request and response rings do not overlap",
                         queue.response_base >= queue.request_base + (FAKE_ENTRIES * 32u),
                         "0x%08x vs 0x%08x", queue.response_base,
                         queue.request_base + (FAKE_ENTRIES * 32u));
}

static void stage_message_encodings(void)
{
    ft_tt_udrv_arc_msg msg;
    uint16_t settings[14];
    unsigned i;

    ft_tt_test_step("Stage 4 — the exact words tt-kmd puts on the wire");

    (void)ft_tt_udrv_arc_msg_asic_state0(&msg);
    ft_tt_test_check_fmt("ASIC_STATE0 is header 0xA0 with an empty payload",
                         msg.header == 0xA0u && msg.payload[0] == 0u && msg.payload[6] == 0u,
                         "0x%08x", msg.header);

    (void)ft_tt_udrv_arc_msg_asic_state3(&msg);
    ft_tt_test_check_fmt("ASIC_STATE3 is header 0xA3", msg.header == 0xA3u, "0x%08x", msg.header);

    (void)ft_tt_udrv_arc_msg_test(&msg);
    ft_tt_test_check_fmt("TEST is header 0x90", msg.header == 0x90u, "0x%08x", msg.header);

    (void)ft_tt_udrv_arc_msg_set_wdt_timeout(10000u, &msg);
    ft_tt_test_check_fmt("SET_WDT_TIMEOUT is 0xC1 with the timeout in payload[0]",
                         msg.header == 0xC1u && msg.payload[0] == 10000u, "0x%08x / %u",
                         msg.header, msg.payload[0]);
    ft_tt_test_check_fmt("...and 10000 ms is the driver's own default (10 s)",
                         msg.payload[0] == 10u * 1000u, "%u", msg.payload[0]);

    (void)ft_tt_udrv_arc_msg_trigger_reset(3u, &msg);
    ft_tt_test_check_fmt("TRIGGER_RESET is 0x56 with argument 3 (ASIC + M3)",
                         msg.header == 0x56u && msg.payload[0] == 3u, "0x%08x / %u", msg.header,
                         msg.payload[0]);

    /* THE ONE THAT MATTERS MOST: this is what sets the clock at open. */
    (void)ft_tt_udrv_arc_msg_power_setting_open_default(&msg);
    ft_tt_test_check_fmt("POWER_SETTING at open is header 0x7FFE0F21", msg.header == 0x7FFE0F21u,
                         "0x%08x", msg.header);
    ft_tt_test_check_fmt("...i.e. every feature ON except MAX_AI_CLK",
                         ((msg.header >> 16) & 0xFFFFu) ==
                             (FT_TT_UDRV_ARC_POWER_FLAG_ALL &
                              ~FT_TT_UDRV_ARC_POWER_FLAG_MAX_AI_CLK),
                         "flags=0x%04x", (msg.header >> 16) & 0xFFFFu);
    ft_tt_test_check_fmt("...with validity 0x0F (15 flags, 0 settings)",
                         ((msg.header >> 8) & 0xFFu) == 0x0Fu, "0x%02x",
                         (msg.header >> 8) & 0xFFu);

    (void)ft_tt_udrv_arc_msg_power_setting_last_close(&msg);
    ft_tt_test_check_fmt("POWER_SETTING at last close is header 0x00000F21 — everything off",
                         msg.header == 0x00000F21u, "0x%08x", msg.header);

    for (i = 0u; i < 14u; i++)
        settings[i] = (uint16_t)(0x1000u + i);

    (void)ft_tt_udrv_arc_msg_power_setting(0x2Fu, 0x0005u, settings, &msg);
    ft_tt_test_check_fmt("a settings payload packs two u16 per word, little-endian first",
                         msg.payload[0] == 0x10011000u && msg.payload[6] == 0x100D100Cu,
                         "%08x .. %08x", msg.payload[0], msg.payload[6]);
    ft_tt_test_check_fmt("...and validity/flags land in bytes 1 and 2-3",
                         msg.header == (0x21u | (0x2Fu << 8) | (0x0005u << 16)), "0x%08x",
                         msg.header);

    ft_tt_test_check("a NULL output is refused",
                     ft_tt_udrv_arc_msg_asic_state0(NULL) == FT_TT_UDRV_EINVAL, "");
}

static void stage_push_pop(void)
{
    fake_arc arc;
    ft_tt_udrv_arc_transport transport;
    ft_tt_udrv_arc_queue queue;
    ft_tt_udrv_arc_msg msg;
    uint32_t index;

    ft_tt_test_step("Stage 5 — push and pop against the simulated ARC");

    fake_init(&arc, &transport);
    (void)ft_tt_udrv_arc_queue_init(FAKE_QUEUE_BASE, FAKE_ENTRIES, &queue);

    arc.respond = false; /* the doorbell does nothing, so the ring just fills */

    (void)ft_tt_udrv_arc_msg_asic_state0(&msg);
    ft_tt_test_check("the first push succeeds",
                     ft_tt_udrv_arc_try_push(&transport, &queue, &msg) == FT_TT_UDRV_OK, "");

    ft_tt_test_check_fmt("the request landed in slot 0",
                         arc.memory[fake_word_index(queue.request_base)] == 0xA0u, "0x%08x",
                         arc.memory[fake_word_index(queue.request_base)]);
    ft_tt_test_check_fmt("...and the write pointer advanced to 1",
                         arc.memory[fake_word_index(queue.req_wptr_addr)] == 1u, "%u",
                         arc.memory[fake_word_index(queue.req_wptr_addr)]);

    ft_tt_test_check("popping an empty response ring is EAGAIN, not an error",
                     ft_tt_udrv_arc_try_pop(&transport, &queue, &msg) == FT_TT_UDRV_EAGAIN, "");

    for (index = 1u; index < FAKE_ENTRIES; index++)
        (void)ft_tt_udrv_arc_try_push(&transport, &queue, &msg);

    ft_tt_test_check("a full request ring refuses with EAGAIN",
                     ft_tt_udrv_arc_try_push(&transport, &queue, &msg) == FT_TT_UDRV_EAGAIN,
                     "4 of 4 slots occupied");

    /* A dead device must be diagnosed as one, not treated as a wild pointer. */
    arc.dead = true;
    ft_tt_test_check("an all-ones pointer read is EIO, not arithmetic",
                     ft_tt_udrv_arc_try_push(&transport, &queue, &msg) == FT_TT_UDRV_EIO,
                     "0xFFFFFFFF means the NoC is gone");
    ft_tt_test_check("...and the same on the pop side",
                     ft_tt_udrv_arc_try_pop(&transport, &queue, &msg) == FT_TT_UDRV_EIO, "");
}

static void stage_send_sync(void)
{
    fake_arc arc;
    ft_tt_udrv_arc_transport transport;
    ft_tt_udrv_arc_queue queue;
    ft_tt_udrv_arc_msg msg;
    uint32_t round;

    ft_tt_test_step("Stage 6 — the synchronous exchange");

    fake_init(&arc, &transport);
    (void)ft_tt_udrv_arc_queue_init(FAKE_QUEUE_BASE, FAKE_ENTRIES, &queue);

    (void)ft_tt_udrv_arc_msg_set_wdt_timeout(10000u, &msg);
    ft_tt_test_check("a well-formed exchange succeeds",
                     ft_tt_udrv_arc_send_sync(&transport, &queue, &msg, 8u) == FT_TT_UDRV_OK, "");
    ft_tt_test_check_fmt("the firmware saw the message we built",
                         arc.last_request.header == 0xC1u && arc.last_request.payload[0] == 10000u,
                         "0x%08x / %u", arc.last_request.header, arc.last_request.payload[0]);
    ft_tt_test_check_fmt("the doorbell was rung exactly once", arc.triggers == 1u, "%u",
                         arc.triggers);
    ft_tt_test_check_fmt("the response overwrote the request, as the driver does",
                         msg.header == 0u && msg.payload[0] == (10000u ^ 0xA5A5A5A5u),
                         "0x%08x / 0x%08x", msg.header, msg.payload[0]);

    /* Round-trip the ring several times so a wrap bug cannot hide behind a
     * single exchange. */
    for (round = 0u; round < 11u; round++) {
        (void)ft_tt_udrv_arc_msg_asic_state0(&msg);
        if (ft_tt_udrv_arc_send_sync(&transport, &queue, &msg, 8u) != FT_TT_UDRV_OK)
            break;
    }
    ft_tt_test_check_fmt("11 more exchanges wrap both rings without wedging", round == 11u,
                         "completed %u of 11", round);

    /* A firmware error must be surfaced AND its status preserved. */
    fake_init(&arc, &transport);
    arc.status = 0xDEADu;
    (void)ft_tt_udrv_arc_msg_asic_state0(&msg);
    ft_tt_test_check("a nonzero response header is a failure",
                     ft_tt_udrv_arc_send_sync(&transport, &queue, &msg, 8u) == FT_TT_UDRV_EIO,
                     "");
    ft_tt_test_check_fmt("...and the firmware's status is left where the caller can read it",
                         msg.header == 0xDEADu, "0x%08x", msg.header);

    /* No response: bounded, and it says EAGAIN rather than hanging. */
    fake_init(&arc, &transport);
    arc.respond = false;
    (void)ft_tt_udrv_arc_msg_asic_state0(&msg);
    ft_tt_test_check("a silent firmware exhausts the poll budget",
                     ft_tt_udrv_arc_send_sync(&transport, &queue, &msg, 5u) == FT_TT_UDRV_EAGAIN,
                     "");
    ft_tt_test_check_fmt("...having waited exactly the budget", arc.waits == 5u, "%u waits",
                         arc.waits);

    /* A stale response from an abandoned exchange must be drained, not
     * mistaken for this exchange's answer. */
    fake_init(&arc, &transport);
    arc.respond = false;
    (void)ft_tt_udrv_arc_msg_asic_state0(&msg);
    (void)ft_tt_udrv_arc_try_push(&transport, &queue, &msg);
    arc.respond = true;
    (void)fake_trigger(&arc); /* firmware answers the ABANDONED request */
    arc.responses_pushed = 0u;

    (void)ft_tt_udrv_arc_msg_set_wdt_timeout(1234u, &msg);
    ft_tt_test_check("a stale response is drained before the new request",
                     ft_tt_udrv_arc_send_sync(&transport, &queue, &msg, 8u) == FT_TT_UDRV_OK, "");
    ft_tt_test_check_fmt("...and the answer belongs to the NEW request",
                         msg.payload[0] == (1234u ^ 0xA5A5A5A5u), "0x%08x", msg.payload[0]);

    ft_tt_test_check("a transport missing an operation is refused",
                     ft_tt_udrv_arc_send_sync(NULL, &queue, &msg, 1u) == FT_TT_UDRV_EINVAL, "");
}

int main(void)
{
    stage_queue_location();
    stage_ring_arithmetic();
    stage_slot_addresses();
    stage_message_encodings();
    stage_push_pop();
    stage_send_sync();

    return ft_tt_test_finish();
}
