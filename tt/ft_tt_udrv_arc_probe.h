/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_arc_probe.h
 * @brief The ARC probe sequence for the USERSPACE DRIVER — the two messages
 *   `blackhole_init_hardware()` sends at probe, plus the `POWER_SETTING` that
 *   `tt_cdev_open()` sends and that nothing of ours has ever sent.
 *
 *   `ft_tt_udrv_arc_msg.[ch]` owns the wire format and the ring arithmetic and
 *   is pure. This file is the two things it could not be: a TRANSPORT bound to
 *   a real NoC path, and the ORDERED SEQUENCE that brings a card up.
 *
 * ---------------------------------------------------------------------------
 * THE THREE ADDRESSES, AND WHICH OF THEM THE CSM BOUND APPLIES TO
 * ---------------------------------------------------------------------------
 * tt-kmd reaches the ARC through two different accessors and the difference
 * matters, because copying the wrong one refuses a legal access or permits an
 * illegal one:
 *
 *   noc_read32/noc_write32   ANY tile-local address on (8, 0). Used for the
 *                            boot status and QCB pointer in the reset scratch
 *                            file at 0x8003xxxx, and for the doorbell FIFO at
 *                            0x800B0000. Neither is in the CSM.
 *   csm_read32/csm_write32   the same NoC access with a bound: the address must
 *                            lie inside [0x10000000, 0x10080000). Used for the
 *                            queue header and both message rings.
 *
 * So the transport vtable this file installs enforces the CSM bound — that is
 * what `ft_tt_udrv_arc_msg` is entitled to assume — and the locate path and the
 * doorbell use the raw accessors below, exactly as the driver does.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES A RESPONSE EVIDENCE OF FIRMWARE, AND NOT OF OUR OWN RING
 * ---------------------------------------------------------------------------
 * The response read pointer is ours; the response WRITE pointer is the ARC's.
 * `ft_tt_udrv_arc_try_pop()` returns EAGAIN until `res_wptr` moves, and nothing
 * on the host can move it. A completed exchange is therefore firmware action,
 * not ring bookkeeping — which is the difference between proving the protocol
 * and proving the card. The red arm that shows this is a transport whose
 * `trigger` does nothing: the request is pushed identically and no response
 * ever arrives.
 *
 * @status ON SILICON 2026-08-17 (p150a at 0000:12:00.0, tt-kmd loaded as the
 *   oracle) via `tests/test_udrv_card_arc.c`. ASIC_STATE0, SET_WDT_TIMEOUT and
 *   POWER_SETTING all answered by firmware; POWER_SETTING's effect confirmed on
 *   an instrument that is not this one (the AI clock, read through tt-kmd's
 *   telemetry and through our own NoC read of the ARC telemetry table).
 * @issues
 *   - ASIC_STATE3 and TRIGGER_RESET are built by `ft_tt_udrv_arc_msg.h` and are
 *     deliberately NOT sequenced here. A power-down at the wrong moment and an
 *     ASIC+DMC reset are both card-wide, and this file is what a bring-up runs.
 * @todo None
 */

#ifndef FT_TT_UDRV_ARC_PROBE_H
#define FT_TT_UDRV_ARC_PROBE_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_udrv_arc_msg.h"
#include "ft_tt_udrv_noc.h"

/** tt-kmd's own `auto_reset_timeout` default, 10 s, in the milliseconds the
 *  message carries (`blackhole_init_hardware()`). */
#define FT_TT_UDRV_ARC_WATCHDOG_DEFAULT_MS 10000u

/** How long to wait for `ARC_BOOT_STATUS` bit 0, matching ARC_MSG_READY_MS. */
#define FT_TT_UDRV_ARC_READY_TIMEOUT_MS 500u

/** Poll budget for one exchange. Each poll sleeps FT_TT_UDRV_ARC_POLL_WAIT_US,
 *  so this is the same order as tt-kmd's 1000 ms jiffies deadline while staying
 *  a COUNT — a deadline cannot be reproduced in a test and a count can. */
#define FT_TT_UDRV_ARC_DEFAULT_POLLS 8000u

/** Between polls, matching the kernel's usleep_range(100, 200). */
#define FT_TT_UDRV_ARC_POLL_WAIT_US 120u

/* ================================================================
 * The transport
 * ================================================================ */

/**
 * A transport context: one NoC path this process owns, aimed wherever the ARC
 * tile is, plus the access counters that let a caller show a red arm really did
 * fewer of something.
 *
 * The path must be UNCACHED (`write_combine = false` at open). Every access
 * here is a register or a ring pointer, and write combining is free to merge
 * and reorder stores inside a buffer — which for a ring pointer published after
 * its payload is not a performance question, it is a corrupt message.
 */
typedef struct ft_tt_udrv_arc_noc_context {
    ft_tt_udrv_noc_path *path;
    uint16_t x; /**< TRANSLATED ARC coordinate. Never probed, never defaulted */
    uint16_t y;

    uint64_t reads;    /**< 32-bit reads issued through this context */
    uint64_t writes;   /**< 32-bit writes issued through this context */
    uint64_t triggers; /**< doorbell writes issued through this context */
} ft_tt_udrv_arc_noc_context;

/**
 * Bind a transport vtable to a NoC path.
 *
 * @param context        caller-owned, zeroed and filled; must outlive the transport
 * @param path           an open, UNCACHED path on a window this process owns
 * @param x, y           TRANSLATED coordinate of the ARC tile, derived by the caller
 * @param out_transport  filled with csm_read32 / csm_write32 / trigger / wait
 * @return               FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL
 *
 * Thread-safe: no (the path is stateful)   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_arc_transport_init(ft_tt_udrv_arc_noc_context *context, ft_tt_udrv_noc_path *path,
                                  uint16_t x, uint16_t y,
                                  ft_tt_udrv_arc_transport *out_transport);

/**
 * A 32-bit read of ANY tile-local address on the ARC tile, with no CSM bound.
 * This is tt-kmd's `noc_read32`, and the only legitimate users of it are the
 * reset scratch file and the doorbell — everything else must go through the
 * bounded accessors in the transport.
 *
 * @return FT_TT_UDRV_OK, FT_TT_UDRV_EINVAL, or FT_TT_UDRV_EIO
 */
int ft_tt_udrv_arc_noc_read32(ft_tt_udrv_arc_noc_context *context, uint32_t address,
                              uint32_t *out_value);

/** Write half of ft_tt_udrv_arc_noc_read32(). */
int ft_tt_udrv_arc_noc_write32(ft_tt_udrv_arc_noc_context *context, uint32_t address,
                               uint32_t value);

/* ================================================================
 * Locating the queue
 * ================================================================ */

/** Everything the locate step read, kept so a caller can print the two
 *  addresses this round exists to confirm rather than assert them. */
typedef struct ft_tt_udrv_arc_locate_report {
    uint32_t boot_status;      /**< RESET_SCRATCH(2) once bit 0 was set */
    uint32_t boot_status_polls;/**< how many reads it took; 0 means it was ready */
    uint32_t qcb_pointer;      /**< the value RESET_SCRATCH(11) holds */
    uint32_t qcb_word0;        /**< queue base */
    uint32_t qcb_word1;        /**< low byte is the entry count */
    /** Which step did not answer, or NULL on success. Without it, a NoC read that
     *  FAILED and an ARC that never became ready produce the SAME report — both
     *  leave `boot_status` and `boot_status_polls` at 0, which reads as a sleeping
     *  ARC and sends the reader after the firmware. On a second card host's cold bring-up the
     *  cause was neither: memory decode was off, the window aim's own readback
     *  refused, and no read ever reached the card. */
    const char *failure_stage;
} ft_tt_udrv_arc_locate_report;

/**
 * tt-kmd's `blackhole_arc_msg_locate_queue()`: wait for the ARC to say it will
 * accept queue operations, read the QCB address out of RESET_SCRATCH(11), read
 * the two QCB words, and build a validated queue descriptor.
 *
 * Stricter than the driver in one place that matters: the QCB pointer itself is
 * bounds-checked before it is dereferenced. tt-kmd passes it straight to
 * `csm_read32`, which refuses out-of-range addresses but reports only -EIO, so
 * "firmware has not published a queue yet" and "the NoC is hung" are the same
 * error there and are distinguishable here.
 *
 * @param context     an initialised transport context
 * @param timeout_ms  how long to wait for ARC_BOOT_STATUS bit 0
 * @param out_report  may be NULL; filled with every word read
 * @param out_queue   the validated queue
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_EIO when the NoC answered all
 *                    ones or the ARC never became ready; FT_TT_UDRV_ENODEV when
 *                    the QCB pointer is not a CSM address (no queue published);
 *                    FT_TT_UDRV_EINVAL for a malformed queue or a bad argument
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_arc_locate_queue(ft_tt_udrv_arc_noc_context *context, unsigned timeout_ms,
                                ft_tt_udrv_arc_locate_report *out_report,
                                ft_tt_udrv_arc_queue *out_queue);

/* ================================================================
 * Exchanges
 * ================================================================ */

/** One request and whatever came back, kept whole so a caller can print the
 *  firmware's own words instead of a verdict derived from them. */
typedef struct ft_tt_udrv_arc_exchange {
    uint32_t request_header;
    uint32_t request_payload[FT_TT_UDRV_ARC_PAYLOAD_WORDS];
    uint32_t response_header;
    uint32_t response_payload[FT_TT_UDRV_ARC_PAYLOAD_WORDS];
    double elapsed_seconds;
    int rc; /**< the FT_TT_UDRV_* result of the send */
} ft_tt_udrv_arc_exchange;

/**
 * Send one request and record the response.
 *
 * A nonzero response header is the firmware's own status and is NOT rewritten:
 * `rc` reports FT_TT_UDRV_EIO and `response_header` holds the status, which is
 * the same split tt-kmd makes when it returns -EREMOTEIO.
 *
 * @param max_polls  0 selects FT_TT_UDRV_ARC_DEFAULT_POLLS
 * @return           the same value left in `out_exchange->rc`
 *
 * Thread-safe: no (the ring is one shared resource per card; the caller
 *   serialises, as the kernel's arc_msg_mutex does)
 * Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_arc_exchange_send(const ft_tt_udrv_arc_transport *transport,
                                 const ft_tt_udrv_arc_queue *queue,
                                 const ft_tt_udrv_arc_msg *request, uint32_t max_polls,
                                 ft_tt_udrv_arc_exchange *out_exchange);

/**
 * The POWER_SETTING a userspace driver must send, because nothing else will.
 *
 * tt-kmd's legacy open asks for `FT_TT_KMD_POWER_FLAGS_ALL & ~MAX_AI_CLOCK` — every power
 * feature on and the AI clock at its MINIMUM ("legacy client: default to
 * AICLK=Low"). That request is the whole reason this card ran at 800 MHz until
 * 2026-08-15. Ours asks for the same feature set with MAX_AI_CLK set, which is
 * what `FT_TT_AI_CLK` unset means everywhere else in this backend.
 *
 * @param max_ai_clock  true for 0x7FFF, false for tt-kmd's own 0x7FFE
 */
int ft_tt_udrv_arc_power_request_build(bool max_ai_clock, ft_tt_udrv_arc_msg *out_msg);

/* ================================================================
 * The sequence
 * ================================================================ */

/** What a bring-up did, message by message. */
typedef struct ft_tt_udrv_arc_probe_report {
    ft_tt_udrv_arc_locate_report locate;
    ft_tt_udrv_arc_exchange asic_state0;
    ft_tt_udrv_arc_exchange set_wdt_timeout;
    ft_tt_udrv_arc_exchange power_setting;
} ft_tt_udrv_arc_probe_report;

/**
 * The whole hardware footprint of `blackhole_init_hardware()` plus the open-time
 * `POWER_SETTING`, in the driver's own order.
 *
 * MRRS is deliberately absent. It lives in the PCI Express capability's DEVCTL
 * register, and `ft_tt_udrv_reset.[ch]`'s config snapshot and restore already
 * rewrites it — a second writer would be a second place to get it wrong.
 * `fw_log_setup()` is absent by decision (ft_tt_udrv.h §4).
 *
 * SET_WDT_TIMEOUT is sent EXPLICITLY rather than inherited, matching the driver,
 * because a previous owner's value is not knowable and the failure mode of the
 * wrong one is the card resetting underneath us.
 *
 * @param transport     one from ft_tt_udrv_arc_transport_init(); the queue is
 *                      located through its own context, so a caller cannot pair
 *                      a transport with a queue read through a different path
 * @param watchdog_ms   0 selects FT_TT_UDRV_ARC_WATCHDOG_DEFAULT_MS
 * @param max_ai_clock  see ft_tt_udrv_arc_power_request_build()
 * @param out_report    may be NULL; filled whether the sequence succeeded or not
 * @return              FT_TT_UDRV_OK when every message was answered with
 *                      status 0; otherwise the first failure's code, with the
 *                      report showing which message it was
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_arc_probe_hardware(const ft_tt_udrv_arc_transport *transport, uint32_t watchdog_ms,
                                  bool max_ai_clock, ft_tt_udrv_arc_probe_report *out_report);

#endif /* FT_TT_UDRV_ARC_PROBE_H */
