/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_watchdog.h
 * @brief The thing that has to exist BEFORE the first resident kernel: a bounded,
 *   progress-based wait over a core group, a dump that says which core is stuck and
 *   whether it is alive, and a shutdown that retires a blocked kernel by ASKING.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS FIRST AND NOT ALONGSIDE
 * ---------------------------------------------------------------------------
 * On this host a hang is not a failed test. It is a physical power cycle: a wedged NoC
 * survives `rmmod`, both `RESET_DEVICE` flavours and a PCI rescan, and only comes back at
 * power-on (spec §7.3, measured 2026-08-11). A resident kernel is a kernel that BLOCKS,
 * so this phase makes that failure the steady state. The watchdog is what keeps the
 * iteration loop in seconds instead of minutes-plus-a-walk-to-the-machine, and it is
 * worth nothing if it is written after the thing it has to catch.
 *
 * ---------------------------------------------------------------------------
 * ALIVE IS NOT THE SAME AS PROGRESSING, AND THE DUMP HAS TO SAY WHICH
 * ---------------------------------------------------------------------------
 * The resident kernel bumps a HEARTBEAT on every poll and a COMPLETED_COUNT only after a
 * descriptor's work is done. Those two moving independently is the whole diagnosis:
 *
 *   heartbeat moving, completed still   → the kernel is alive and BLOCKED (a semaphore
 *                                          that will never be posted, an empty queue).
 *                                          Recoverable by asking it to stop.
 *   heartbeat still                      → the core is not executing our loop at all.
 *                                          Read the mailbox: it is a launch that never
 *                                          landed, a wedged core, or a kernel that
 *                                          returned.
 *
 * `ft_tt_launch_wait_done_all()` can say neither, because `RUN_MSG_DONE` is 0 and is what
 * a core's own boot leaves behind — it once reported eight cores complete with one never
 * sent a go message (dev notes §3). Nothing in this file waits on a run message.
 *
 * ---------------------------------------------------------------------------
 * THE SHUTDOWN PROTOCOL — AND WHAT IT REFUSES TO DO
 * ---------------------------------------------------------------------------
 * `ft_tt_watchdog_shutdown()` writes `stop_request` and waits for the kernel to reach
 * FT_TT_QUEUE_STATE_EXIT. It does NOT assert reset. Killing a kernel blocked in a wait is
 * how Tensix 0 was wedged on 2026-08-12 — every later program on it hung in
 * `tensix_sync()` until a host reboot (dev notes §1) — and in this phase every resident
 * kernel is blocked in a wait by design. If the stop is not honoured, this reports and
 * leaves the core exactly as it is: recovery is then the operator's decision with the
 * evidence in front of them, which is the same shape as `ft_tt_launch_wait_done()`'s
 * timeout path.
 *
 * @status Green. Caught a deliberate deadlock on the card in 500 ms, reported the core as
 *   ALIVE-but-blocked, and retired it by writing one word with the card still usable
 *   afterwards (docs/tt_bringup_log.md 17.4). Extended for the resident TRIAD: retirement
 *   now requires ALL FIVE processors to report EXIT, the ALIVE verdict comes from the
 *   per-processor heartbeats rather than the reader's alone, and a stall names WHICH
 *   processor is in WHICH wait.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_WATCHDOG_H
#define FT_TT_WATCHDOG_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_group.h"
#include "ft_tt_launch.h"
#include "ft_tt_queue.h"

/** How the wait is bounded. Two bounds, because they catch different things: a STALL is
 *  "nothing finished for a while", which a long op legitimately looks like; the OVERALL
 *  bound is the one that guarantees the call returns. */
typedef struct ft_tt_watchdog_config {
    uint32_t stall_timeout_ms;   /**< no core's completed_count advanced for this long */
    uint32_t overall_timeout_ms; /**< hard bound on the whole wait; 0 is refused */
} ft_tt_watchdog_config;

/** What a wait found. Filled on success and on failure — a timeout with no state is the
 *  thing that made a session guess. */
typedef struct ft_tt_watchdog_report {
    uint32_t cores;              /**< cores in the group */
    uint32_t completed_total;    /**< sum of completed_count */
    uint32_t completed_target;   /**< what it was waiting for, summed */
    uint32_t stalled_slot;       /**< first core short of its target */
    bool stalled_core_is_alive;  /**< its heartbeat moved during the stall window */
    uint32_t stalled_state;      /**< its FT_TT_QUEUE_FIELD_STATE */
    uint32_t stalled_error;      /**< its FT_TT_QUEUE_FIELD_ERROR */
    uint32_t elapsed_ms;
    /** WHICH PROCESSOR, for a group whose `resident_processor_mask` is set. A triad that
     *  is stuck is stuck in one of five places, and "the core is blocked" does not say
     *  which — the first processor found not ticking, or the first still short of the
     *  others' cursor. FT_TT_PROC_COUNT means "the mask was 0, so this is a single-processor
     *  program and the control block already said everything". */
    uint32_t stalled_processor;
    uint32_t stalled_processor_state;  /**< that processor's FT_TT_QUEUE_STATUS_STATE */
    uint32_t stalled_processor_cursor; /**< and the descriptor it is on */
} ft_tt_watchdog_report;

/**
 * Wait for every core in the group to reach `target_per_core` completed descriptors.
 *
 * Progress-based: the stall clock resets whenever ANY core's completed_count advances, so
 * a group doing slow work is not mistaken for a group that is stuck, while a group that
 * is genuinely stuck is reported after `stall_timeout_ms` rather than after the whole
 * overall bound.
 *
 * On failure the group is DUMPED before this returns, and nothing is reset.
 *
 * @param out_report  may be NULL
 * @return            FT_TT_OK; FT_TT_EIO on stall or overall timeout; FT_TT_EINVAL
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_watchdog_wait(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                        uint32_t target_per_core, const ft_tt_watchdog_config *config,
                        ft_tt_watchdog_report *out_report);

/**
 * Print every core's queue control block AND its mailbox state to stderr.
 *
 * Reads only, so it is safe against a core that is stuck — which is when it is wanted.
 * The queue half says whether the kernel is alive and what it is waiting for; the mailbox
 * half (ft_tt_launch_dump_state) says whether the firmware ever handed control to it.
 * Both, because a session already spent a day on a fault the wrong half could not see.
 */
void ft_tt_watchdog_dump(const ft_tt_noc_access *noc, const ft_tt_core_group *group);

/**
 * Retire every resident kernel in the group by ASKING.
 *
 * 1. `stop_request = 1` on every core — one L1 word, seen by the idle poll and by every
 *    blocking opcode.
 * 2. Wait for each core to reach FT_TT_QUEUE_STATE_EXIT, bounded.
 * 3. Wait for `RUN_MSG_DONE`, which is now meaningful because step 2 already proved the
 *    kernel itself finished — the run message is confirmation, not evidence.
 *
 * It does NOT park. The core is left with its firmware up and ready for the next launch,
 * because "retire the resident kernel" and "stop using this core" are different
 * decisions: a parked core needs its firmware re-booted before it can run anything, and
 * folding the two together makes every shutdown a reboot. Callers park on their own exit
 * path, once.
 *
 * A core that does not honour the stop is REPORTED and left alone. There is no reset
 * escalation in this function and there must not be one added: see the header note.
 *
 * @param out_unstopped  may be NULL; count of cores that did not reach EXIT
 * @return               FT_TT_OK; FT_TT_EIO if any core failed to stop
 */
int ft_tt_watchdog_shutdown(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                            uint32_t timeout_ms, uint32_t *out_unstopped);

#endif /* FT_TT_WATCHDOG_H */
