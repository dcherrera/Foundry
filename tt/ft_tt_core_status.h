/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_status.h
 * @brief Read-only per-Tensix state: the soft-reset register decoded per processor, the
 *   mailbox run signals, `subordinate_sync`, any resident-queue control block, and a
 *   one-word verdict.
 *
 * **This module issues READS ONLY.** That is what makes it safe to point at a core in an
 * unknown state, which is the only time anyone wants it.
 *
 * **And a read can only ever be half the answer.** A wedged Tensix answers NoC reads
 * exactly like a healthy one — that is precisely what made the first status checks on
 * this project lie (dev notes §3). Everything here is a description of what the core's
 * registers SAY; whether the core still computes is `ft_tt_core_probe.h`, which boots
 * firmware and runs real arithmetic. The verdicts below are named so that no reading of
 * them claims more than a read can support: a core parked with all five RISC-Vs held
 * comes back HELD, never "healthy".
 *
 * @status Runs on the p150a test card. Sweeps 140 cores in a few seconds. MEASURED
 *   2026-08-12: three wedged cores read IDENTICALLY to seven healthy ones (`0x00047800`,
 *   all five held) — the verdicts here are worded around exactly that.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CORE_STATUS_H
#define FT_TT_CORE_STATUS_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_coord.h"
#include "ft_tt_launch.h"
#include "ft_tt_noc.h"
#include "ft_tt_queue.h"

/** What one read of a core's registers supports saying about it, and nothing more. */
typedef enum ft_tt_core_verdict {
    /** The core did not answer a mailbox read. Past what a host write can fix. */
    FT_TT_CORE_VERDICT_NO_ANSWER = 0,
    /** Every RISC-V held in reset. The normal cold/parked state — and also what a
     *  wedged core looks like after someone parked it. A read cannot separate the two;
     *  only a probe can. */
    FT_TT_CORE_VERDICT_HELD,
    /** BRISC running, go signal DONE: firmware is up and waiting for a launch message. */
    FT_TT_CORE_VERDICT_FIRMWARE_IDLE,
    /** BRISC running with the go signal still at INIT: the boot handshake was started and
     *  never finished. On a core that was released some time ago this is a STUCK core, and
     *  it is the one wedge shape a read alone can name. */
    FT_TT_CORE_VERDICT_BOOT_INCOMPLETE,
    /** Go signal is GO: a kernel is executing, or blocked. */
    FT_TT_CORE_VERDICT_KERNEL_RUNNING,
    /** BRISC running with subordinates HELD in reset and their sync bytes not DONE —
     *  the core-60 signature. BRISC is inside `tensix_sync()` waiting for processors
     *  that are in reset and can never answer. */
    FT_TT_CORE_VERDICT_SYNC_BLOCKED,
    /** Some RISC-Vs running, some held, and nothing above matches. Report, do not guess. */
    FT_TT_CORE_VERDICT_MIXED
} ft_tt_core_verdict;

/** Highest verdict value, for sizing a tally array. */
#define FT_TT_CORE_VERDICT_LAST FT_TT_CORE_VERDICT_MIXED

/** One core's readable state. Every field comes from a NoC read; nothing is inferred. */
typedef struct ft_tt_core_status {
    uint32_t index;         /**< Tensix index, as `ft_tt_coord_tensix_by_index` numbers them */
    uint16_t x;             /**< TRANSLATED NoC coordinate, from live ARC telemetry */
    uint16_t y;
    int read_result;        /**< FT_TT_OK, or why the mailbox read failed */
    ft_tt_launch_state state;

    bool queue_present;     /**< a resident-queue control block with our magic was found */
    uint32_t queue_control[FT_TT_QUEUE_FIELD_COUNT];
    bool queue_heartbeat_ticking; /**< the control heartbeat changed between two reads */

    ft_tt_core_verdict verdict;
} ft_tt_core_status;

/** "no answer", "held", "firmware idle", ... Never NULL. */
const char *ft_tt_core_verdict_string(ft_tt_core_verdict verdict);

/**
 * Read one core's state and classify it. Issues only NoC reads.
 *
 * @param noc     device access
 * @param grid    live grid, for resolving `index` to a coordinate. NEVER a guess: a
 *                wrong endpoint hangs the chip (spec §7.3)
 * @param index   Tensix index
 * @param out     zeroed and filled; `read_result` carries a failed mailbox read rather
 *                than being returned, so a sweep can report the core and continue
 * @return        FT_TT_OK once the coordinate resolved, or the coordinate error
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_core_status_read(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                           uint32_t index, ft_tt_core_status *out);

/**
 * The verdict, as a PURE FUNCTION of `read_result` and `state` — no device access at all.
 *
 * Exported so the classification can be exercised against hand-built register readings on a
 * machine with no card. Every verdict anything now ACTS on — Pantheon's arming path refuses
 * to boot firmware on the strength of these — has a red arm because of it, and a check
 * nobody has watched fail is not evidence (coding-best-practices §9).
 *
 * `ft_tt_core_status_read()` calls it and fills `out->verdict`; a caller that already holds
 * a filled status has no reason to call it again.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
ft_tt_core_verdict ft_tt_core_status_classify(const ft_tt_core_status *status);

/** Is this processor held in reset? `processor` is an FT_TT_PROC_* index. */
bool ft_tt_core_status_processor_is_held(uint32_t reset_register, unsigned processor);

/* The three printers live in tt/tools/ft_tt_core_status_print.h. They were split out when
 * this file became a library file: nothing that a serve links should be able to write to
 * stdout, and the tool that wants them is the only caller they ever had. */

#endif /* FT_TT_CORE_STATUS_H */
