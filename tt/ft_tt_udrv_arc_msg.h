/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_arc_msg.h
 * @brief The ARC firmware message queue — the mechanism behind every hardware
 *   action tt-kmd performs that is not a TLB register write: the power-up
 *   handshake, the watchdog timeout, the clock/power state, and the ASIC+DMC
 *   reset.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS
 * ---------------------------------------------------------------------------
 * A plain message ring in the ARC's CSM, reached by ordinary 32-bit NoC reads
 * and writes to tile (8,0) — the same path `ft_tt_arc_telemetry.c` already
 * uses to read telemetry. Messaging adds WRITES to a path Foundry already
 * reaches, and one write to a doorbell FIFO. There is no ioctl in it anywhere:
 * tt-kmd's `arc_msg_send_sync()` is `csm_read32`/`csm_write32` in a loop, and those are `noc_read32`/`noc_write32` through the
 * driver's own private TLB window.
 *
 *   locate   RESET_SCRATCH(11) = 0x8003042C holds the QCB address.
 *            QCB word 0 = queue base (a CSM address), word 1 & 0xFF = entries.
 *   header   32 bytes at the queue base: req_wptr +0x00, res_rptr +0x04,
 *            req_rptr +0x10, res_wptr +0x14.
 *   request  ring of `num_entries` 32-byte messages at base + 32.
 *   response ring of `num_entries` 32-byte messages after the requests.
 *   trigger  write 0 to 0x800B0000 (ARC_MSI_FIFO) on the same tile.
 *   status   a response whose header word is nonzero is a firmware error.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS PROVEN HERE AND WHAT IS NOT
 * ---------------------------------------------------------------------------
 * Message ENCODING and RING ARITHMETIC are pure functions and are gated
 * off-hardware against an independently written simulated ARC (the test's fake
 * implements the peer side of the protocol from the driver's semantics, not by
 * calling these functions — instrumenting a mechanism with itself proves
 * nothing). The SEND is not proven and cannot be: it needs a NoC transaction.
 * That is why the transport is a vtable — `ft_tt_udrv_arc_transport` — with
 * exactly three operations, so the untested surface is three functions and not
 * the protocol.
 *
 * ---------------------------------------------------------------------------
 * RING OCCUPANCY FOR ANY ENTRY COUNT
 * ---------------------------------------------------------------------------
 * Pointers run over [0, 2N). A single modular subtraction in u32 is only right
 * when 2N divides 2^32 (N a power of two, as on this card's firmware). This
 * module branches on `wptr >= rptr` instead, so it is right for any N, and it
 * refuses pointers outside [0, 2N).
 *
 * @status Still pure — nothing here issues a NoC transaction — but the protocol
 *   above it is NO LONGER UNPROVEN. `ft_tt_udrv_arc_probe.[ch]` supplies a real
 *   transport over a TLB window we aim ourselves, and on 2026-08-17 this ring
 *   carried ASIC_STATE0, SET_WDT_TIMEOUT, POWER_SETTING and TEST to the p150a's
 *   firmware, each answered with status 0. The queue address
 *   (`RESET_SCRATCH(11)`) and the doorbell (0x800B0000) were both confirmed
 *   against tt-kmd on live silicon; see ft_tt_udrv_arc_probe.h.
 * @issues
 *   - **TEST (0x90) returns a firmware-side COUNTER in `payload[1]`**, one per
 *     TEST message whichever path sent it. A caller comparing a whole response
 *     against another one will see it move and must not read that as a
 *     mismatch.
 * @todo None
 */

#ifndef FT_TT_UDRV_ARC_MSG_H
#define FT_TT_UDRV_ARC_MSG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------------ *
 * Addresses and constants, all the kernel driver's documented Blackhole ARC values.
 * ------------------------------------------------------------------------ */

/** NoC coordinates of the ARC tile. Never probed; this is the driver's own. */
#define FT_TT_UDRV_ARC_X 8u
#define FT_TT_UDRV_ARC_Y 0u

/** RESET_SCRATCH(N) — the ARC scratch register file. */
#define FT_TT_UDRV_ARC_RESET_SCRATCH(n) (0x80030400u + ((uint32_t)(n) * 4u))

/** Holds the Message Queue Control Block address. */
#define FT_TT_UDRV_ARC_MSG_QCB_PTR FT_TT_UDRV_ARC_RESET_SCRATCH(11)

/** Boot status; bit 0 means the ARC will accept queue operations. */
#define FT_TT_UDRV_ARC_BOOT_STATUS FT_TT_UDRV_ARC_RESET_SCRATCH(2)
#define FT_TT_UDRV_ARC_BOOT_STATUS_READY_FOR_MSG 0x1u

/** Write 0 here to make the ARC process the request ring. */
#define FT_TT_UDRV_ARC_MSI_FIFO 0x800B0000u

/** The ARC's CSM window. Every queue address must lie inside it. */
#define FT_TT_UDRV_ARC_CSM_BASE 0x10000000u
#define FT_TT_UDRV_ARC_CSM_SIZE (1u << 19)

/** Message types we need. */
#define FT_TT_UDRV_ARC_MSG_ASIC_STATE0 0xA0u     /**< power-up handshake */
#define FT_TT_UDRV_ARC_MSG_ASIC_STATE3 0xA3u     /**< power-down, at cleanup */
#define FT_TT_UDRV_ARC_MSG_POWER_SETTING 0x21u   /**< clock / power features */
#define FT_TT_UDRV_ARC_MSG_SET_WDT_TIMEOUT 0xC1u /**< firmware watchdog */
#define FT_TT_UDRV_ARC_MSG_TRIGGER_RESET 0x56u   /**< ASIC + DMC reset */
#define FT_TT_UDRV_ARC_MSG_TEST 0x90u            /**< liveness probe */

/** Power feature bits, `power_flags`. Bit 15 is reserved by the FW spec. */
#define FT_TT_UDRV_ARC_POWER_FLAG_MAX_AI_CLK (1u << 0)
#define FT_TT_UDRV_ARC_POWER_FLAG_MRISC_PHY_WAKEUP (1u << 1)
#define FT_TT_UDRV_ARC_POWER_FLAG_TENSIX_ENABLE (1u << 2)
#define FT_TT_UDRV_ARC_POWER_FLAG_L2CPU_ENABLE (1u << 3)
#define FT_TT_UDRV_ARC_POWER_FLAG_ALL 0x7FFFu

/** Bytes per ring entry, and the queue header that precedes both rings. */
#define FT_TT_UDRV_ARC_MSG_BYTES 32u
#define FT_TT_UDRV_ARC_QUEUE_HEADER_BYTES 32u

/** Payload words per message. */
#define FT_TT_UDRV_ARC_PAYLOAD_WORDS 7u

/** A request or a response. Wire layout is the header word then the payload. */
typedef struct ft_tt_udrv_arc_msg {
    uint32_t header;
    uint32_t payload[FT_TT_UDRV_ARC_PAYLOAD_WORDS];
} ft_tt_udrv_arc_msg;

/** A located message queue. All addresses are CSM addresses. */
typedef struct ft_tt_udrv_arc_queue {
    uint32_t base;          /**< queue base, from QCB word 0 */
    uint32_t num_entries;   /**< QCB word 1 & 0xFF */
    uint32_t request_base;  /**< base + 32 */
    uint32_t response_base; /**< request_base + num_entries * 32 */
    uint32_t req_wptr_addr;
    uint32_t req_rptr_addr;
    uint32_t res_wptr_addr;
    uint32_t res_rptr_addr;
} ft_tt_udrv_arc_queue;

/**
 * The three device operations the protocol needs. Everything above this line
 * is arithmetic; everything below it is a NoC transaction.
 *
 * `wait` is called between polls and may be NULL. It exists so the polling
 * loop has no dependency on a clock, which is what makes `send_sync` testable
 * with a deterministic fake.
 */
typedef struct ft_tt_udrv_arc_transport {
    int (*csm_read32)(void *context, uint32_t address, uint32_t *out_value);
    int (*csm_write32)(void *context, uint32_t address, uint32_t value);
    int (*trigger)(void *context);
    void (*wait)(void *context);
    void *context;
} ft_tt_udrv_arc_transport;

/* ------------------------------------------------------------------------ *
 * Pure arithmetic — provable off-hardware.
 * ------------------------------------------------------------------------ */

/**
 * Build a queue descriptor from the two QCB words, validating that the whole
 * ring lies inside the CSM.
 *
 * @param qcb_word0  queue base
 * @param qcb_word1  low byte is the entry count
 * @return           FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL for a NULL argument, a
 *                   zero entry count (which would divide by zero in the ring
 *                   maths — the kernel driver carries the same guard), or
 *                   a ring that does not fit in the CSM
 */
int ft_tt_udrv_arc_queue_init(uint32_t qcb_word0, uint32_t qcb_word1,
                              ft_tt_udrv_arc_queue *out_queue);

/**
 * Correct ring occupancy: `(wptr - rptr) mod 2 * num_entries`, for any entry
 * count, refusing pointers that are already out of range.
 *
 * @return FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL
 */
int ft_tt_udrv_arc_ring_occupancy(uint32_t wptr, uint32_t rptr, uint32_t num_entries,
                                  uint32_t *out_occupancy);

/** Address of request slot `wptr % num_entries`, word `word_index` of 8. */
int ft_tt_udrv_arc_request_word_address(const ft_tt_udrv_arc_queue *queue, uint32_t wptr,
                                        uint32_t word_index, uint32_t *out_address);

/** Address of response slot `rptr % num_entries`, word `word_index` of 8. */
int ft_tt_udrv_arc_response_word_address(const ft_tt_udrv_arc_queue *queue, uint32_t rptr,
                                         uint32_t word_index, uint32_t *out_address);

/** Advance a ring pointer: `(pointer + 1) mod 2 * num_entries`. */
int ft_tt_udrv_arc_ring_advance(uint32_t pointer, uint32_t num_entries, uint32_t *out_pointer);

/** Wire word `word_index` of a message: index 0 is the header. */
int ft_tt_udrv_arc_msg_word(const ft_tt_udrv_arc_msg *msg, uint32_t word_index,
                            uint32_t *out_value);

/* ------------------------------------------------------------------------ *
 * Message builders — the four exchanges the userspace driver owes the card.
 * ------------------------------------------------------------------------ */

/** ASIC_STATE0, no payload. The power-up handshake at probe. */
int ft_tt_udrv_arc_msg_asic_state0(ft_tt_udrv_arc_msg *out_msg);

/** ASIC_STATE3, no payload. What tt-kmd sends at cleanup_hardware. */
int ft_tt_udrv_arc_msg_asic_state3(ft_tt_udrv_arc_msg *out_msg);

/** SET_WDT_TIMEOUT. tt-kmd sends `auto_reset_timeout` seconds x 1000, default
 *  10 s => 10000. Read ft_tt_udrv.h §3 before choosing a different value. */
int ft_tt_udrv_arc_msg_set_wdt_timeout(uint32_t milliseconds, ft_tt_udrv_arc_msg *out_msg);

/**
 * POWER_SETTING. `header = 0x21 | validity << 8 | power_flags << 16`, and the
 * fourteen 16-bit settings pack two per payload word, little-endian — which is
 * a `memcpy` in the driver and is written out explicitly
 * here so it does not depend on host endianness.
 *
 * @param validity        FT_TT_KMD_POWER_VALIDITY: low nibble = number of valid
 *                        flags, high nibble = number of valid settings
 * @param power_flags     FT_TT_UDRV_ARC_POWER_FLAG_*
 * @param settings        14 values; may be NULL for all zero
 */
int ft_tt_udrv_arc_msg_power_setting(uint8_t validity, uint16_t power_flags,
                                     const uint16_t *settings, ft_tt_udrv_arc_msg *out_msg);

/**
 * The exact POWER_SETTING tt-kmd sends when a legacy client opens the char
 * device and no other fd is open: validity 0x0F, flags
 * `FT_TT_UDRV_ARC_POWER_FLAG_ALL & ~MAX_AI_CLK` = 0x7FFE, payload zero.
 * Header 0x7FFE0F21.
 *
 * Note what that means: the driver's default deliberately requests the LOW AI
 * clock ("legacy client: default to AICLK=Low" in the driver's open()). Every timing
 * this project has recorded was taken with that request in force.
 */
int ft_tt_udrv_arc_msg_power_setting_open_default(ft_tt_udrv_arc_msg *out_msg);

/**
 * The POWER_SETTING tt-kmd sends at LAST CLOSE: the aggregation runs over an
 * empty fd list, so every feature is off. Validity 0x0F, flags 0, header
 * 0x00000F21. `blackhole_class` does not set `defer_idle_powerdown`, so this
 * lands synchronously — there is no grace window on this part.
 */
int ft_tt_udrv_arc_msg_power_setting_last_close(ft_tt_udrv_arc_msg *out_msg);

/** TEST, no payload. tt-kmd sends it before an ASIC+DMC reset to prove the
 *  queue is alive; a failure there is the "NOC is likely hung" diagnosis. */
int ft_tt_udrv_arc_msg_test(ft_tt_udrv_arc_msg *out_msg);

/** TRIGGER_RESET. `argument` is 3 for the ASIC + M3 reset tt-kmd performs. */
int ft_tt_udrv_arc_msg_trigger_reset(uint16_t argument, ft_tt_udrv_arc_msg *out_msg);

/* ------------------------------------------------------------------------ *
 * The protocol — pure arithmetic over a caller-supplied transport.
 * ------------------------------------------------------------------------ */

/** Push one request if the ring has room. FT_TT_UDRV_EAGAIN when full. */
int ft_tt_udrv_arc_try_push(const ft_tt_udrv_arc_transport *transport,
                            const ft_tt_udrv_arc_queue *queue, const ft_tt_udrv_arc_msg *msg);

/** Pop one response if there is one. FT_TT_UDRV_EAGAIN when there is not. */
int ft_tt_udrv_arc_try_pop(const ft_tt_udrv_arc_transport *transport,
                           const ft_tt_udrv_arc_queue *queue, ft_tt_udrv_arc_msg *out_msg);

/**
 * Drain stale responses, push, trigger, and poll for the reply — tt-kmd's
 * `arc_msg_send_sync()` without the kernel's mutex and with a poll budget
 * instead of a jiffies deadline, so the loop is deterministic in a test.
 *
 * @param msg        the request; on success it is OVERWRITTEN by the response,
 *                   exactly as the driver does
 * @param max_polls  how many times to look for the response before giving up.
 *                   0 means one look
 * @return           FT_TT_UDRV_OK when a response arrived with header 0;
 *                   FT_TT_UDRV_EIO when the firmware reported a nonzero status
 *                   (the status is left in `msg->header`) or a transport call
 *                   failed; FT_TT_UDRV_EAGAIN on poll exhaustion;
 *                   FT_TT_UDRV_EINVAL on a bad argument
 *
 * Thread-safe: NO. The ring is one shared resource per card and this function
 *   does not serialise access to it — the caller must, exactly as the kernel's
 *   `arc_msg_mutex` does. With several jobs on one card that has to be a
 *   cross-process lock (`ft_tt_udrv_claim.h`), not a mutex.
 * Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_arc_send_sync(const ft_tt_udrv_arc_transport *transport,
                             const ft_tt_udrv_arc_queue *queue, ft_tt_udrv_arc_msg *msg,
                             uint32_t max_polls);

#endif /* FT_TT_UDRV_ARC_MSG_H */
