/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_recover.h
 * @brief The three CORE-level rungs of the recovery ladder, cheapest first. Every one of
 *   them is a NoC write to one Tensix; nothing here touches the device or the PCIe bus.
 *
 * A Tensix that will not boot firmware again is not damaged silicon — every core this
 * card has stuck has come back — but until now the only known way to clear one was a host
 * reboot, and this host also runs eight K80s. These are the cheap things to try first.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAY BE WRITTEN, AND WHAT MAY NOT
 * ---------------------------------------------------------------------------
 * `FT_TT_SOFT_RESET_ALL_RISCS` is the whole of what any of this touches. The Tensix
 * BACKEND's reset bits live in the same register and asserting them silently corrupts
 * that core's compute until the machine is power-cycled — it cost 14 cores once
 * (`ft_tt_launch.h`, dev notes §1). No function in this file names them.
 *
 * Rung 0 DEASSERTS, which cannot corrupt configuration: the worst case is a subordinate
 * running its firmware `main()` and parking in the go-message wait, which is where a
 * healthy one sits anyway. Dev notes §4 warns that deasserting the non-BRISC RISCs races
 * BRISC firmware — true of a healthy boot, and not of a BRISC already blocked waiting for
 * exactly those processors.
 *
 * ---------------------------------------------------------------------------
 * THE RUNGS
 * ---------------------------------------------------------------------------
 *   0  deassert the RISC-Vs only — one register write
 *   1  reload firmware and run the normal boot sequence
 *   2  write the value the core is waiting FOR: the resident queue's stop flag, the
 *      subordinate-sync bytes a BRISC in `tensix_sync()` polls, and the circular-buffer
 *      page counters a blocked CB wait spins on
 *
 * Rungs 3-5 are device- and bus-level and live in `ft_tt_device_reset.h`.
 *
 * @status Runs on the p150a test card. MEASURED 2026-08-12: none of these three rungs
 *   recovered cores 60, 62 or 65 — all three wedges were in the Tensix BACKEND, which is not
 *   executing anything that could poll a flag or a counter, so no host write reaches them.
 *   The rung that did recover them is rung 3 in ft_tt_device_reset.h.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CORE_RECOVER_H
#define FT_TT_CORE_RECOVER_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_core_probe.h"
#include "ft_tt_core_status.h"

/** Which rung a caller asked for. The numbering is the ladder's, and it is stable. */
typedef enum ft_tt_core_recovery_rung {
    FT_TT_RECOVERY_RUNG_DEASSERT_RISCS = 0,
    FT_TT_RECOVERY_RUNG_RELOAD_FIRMWARE = 1,
    FT_TT_RECOVERY_RUNG_POKE_WAIT = 2
} ft_tt_core_recovery_rung;

/** What a rung did, in enough detail to write down. */
typedef struct ft_tt_core_recovery_report {
    uint32_t index;
    uint16_t x;
    uint16_t y;
    ft_tt_core_recovery_rung rung;
    uint32_t reset_register_before;
    uint32_t reset_register_after;
    uint32_t subordinate_sync_before; /**< the four bytes, NCRISC first, as one word */
    uint32_t subordinate_sync_after;
    unsigned circular_buffers_poked;
    bool queue_stop_requested;
    int result;                       /**< FT_TT_OK if the rung's own writes succeeded */
} ft_tt_core_recovery_report;

/** "deassert RISC-Vs", "reload firmware", "poke the wait". Never NULL. */
const char *ft_tt_core_recovery_rung_string(ft_tt_core_recovery_rung rung);

/**
 * Rung 0 — release `FT_TT_SOFT_RESET_ALL_RISCS` on one core, and nothing else.
 *
 * @param noc     device access
 * @param status  a status read taken immediately before, for the coordinate and the
 *                before-state
 * @param out     zeroed and filled
 * @return        FT_TT_OK once the rung ran; a negative code if it could not
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_core_recover_deassert_riscs(const ft_tt_noc_access *noc,
                                      const ft_tt_core_status *status,
                                      ft_tt_core_recovery_report *out);

/**
 * Rung 1 — the normal boot sequence from scratch: assert ALL_RISCS, clear the mailbox,
 * load all five firmware images, prime the handshake with `RUN_MSG_INIT`, deassert BRISC,
 * wait for ready.
 *
 * This is `ft_tt_core_boot_firmware()`, which is also the first thing a probe does — so a
 * core that passes this rung passes the probe's boot for free. It is a separate rung
 * because "the firmware will not boot" is the failure being climbed out of, and a rung
 * that only reloads is worth distinguishing from one that also computes.
 *
 * It begins with `assert_reset`, so it carries the probe's refusal for the same reason: a
 * core whose kernel is still running is not reset from under it without `force`, which is
 * only defensible on a core already known wedged.
 *
 * @param firmware  images to load; the same build the probe will use
 * @param force     override the running-kernel refusal
 */
int ft_tt_core_recover_reload_firmware(const ft_tt_noc_access *noc,
                                       const ft_tt_core_status *status,
                                       const ft_tt_firmware_set *firmware, bool force,
                                       ft_tt_core_recovery_report *out);

/**
 * Rung 2 — write the value the core is waiting for, so its own kernel can reach its own
 * exit. Three pokes, cheapest and most targeted first:
 *
 *   1. If a resident-queue control block is present, set its stop flag. This is the
 *      project's sanctioned retirement path — ask, never reset.
 *   2. Write `RUN_SYNC_MSG_ALL_SUBORDINATES_DONE` into the four subordinate-sync bytes.
 *      A BRISC stuck in `tensix_sync()` polls exactly those bytes, waiting for TRISCs
 *      that are held in reset and can never write them. This is the core-60 signature.
 *   3. For every circular buffer, set `tiles_received = tiles_acked + 64` in the NoC
 *      overlay stream registers, which is what a consumer blocked in `cb_wait_front` or
 *      the stock wait-for-tiles routine spins on. 64 is above any page count this backend
 *      configures, so
 *      one write satisfies any outstanding wait.
 *
 * Poke 3 is deliberately last: it rewrites flow-control state for all 64 CBs, which is
 * only defensible on a core that is already stuck.
 *
 * **What this cannot reach.** A processor stalled in the Tensix BACKEND — an MVMUL
 * waiting on SrcA/SrcB dvalid, with the MATH RISC-V stalled pushing into a full
 * instruction FIFO — is not executing anything that could poll a flag or a counter. No
 * value written from the host reaches it. That is core 62's shape.
 */
int ft_tt_core_recover_poke_wait(const ft_tt_noc_access *noc, const ft_tt_core_status *status,
                                 ft_tt_core_recovery_report *out);

/** Several lines: what was written, and the before/after registers. */
void ft_tt_core_recovery_print(const ft_tt_core_recovery_report *report);

#endif /* FT_TT_CORE_RECOVER_H */
