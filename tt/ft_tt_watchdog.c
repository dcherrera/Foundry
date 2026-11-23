/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_watchdog.c
 * @brief Bounded progress wait, group dump and the ask-don't-reset shutdown. See
 *   ft_tt_watchdog.h for why none of this touches the soft-reset register.
 *
 * @status Green on the card. Drives the deliberate-deadlock case in
 *   tests/test_phase5_resident.c; no reset was needed to recover from it.
 * @issues None
 * @todo None
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX.1-2008; -std=c99 turns off glibc's
 * default feature set, so ask for them by name or the Linux card host's gcc hides them. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_watchdog.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "ft_tt_device.h"

static uint64_t monotonic_microseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0u;
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)(now.tv_nsec / 1000);
}

/* ------------------------------------------------------------------------- */
/* The wait                                                                  */
/* ------------------------------------------------------------------------- */

/** Sum completed_count over the group, and note the first core short of target. */
static int sample_group(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                        uint32_t target_per_core, uint32_t *out_total, uint32_t *out_short_slot,
                        bool *out_all_reached)
{
    unsigned count = ft_tt_launch_core_set_count(&group->cores);
    unsigned slot;
    uint32_t total = 0u;
    bool all = true;

    *out_short_slot = 0u;
    for (slot = 0u; slot < count; slot++) {
        uint32_t completed = 0u;
        int rc = ft_tt_queue_read_field(noc, &group->queues[slot],
                                        FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &completed);
        if (rc != FT_TT_OK)
            return rc;
        total += completed;
        if (completed < target_per_core && all) {
            all = false;
            *out_short_slot = slot;
        }
    }
    *out_total = total;
    *out_all_reached = all;
    return FT_TT_OK;
}

/**
 * Is ANY resident processor's heartbeat moving?
 *
 * Two reads a moment apart, per processor, and the first one that moves answers yes. This
 * is the ALIVE-versus-NOT-TICKING verdict for a triad, and it has to be per processor
 * because the control block's heartbeat belongs to the reader alone.
 */
static bool any_processor_ticking(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                                  const ft_tt_queue *stuck)
{
    unsigned processor;

    for (processor = 0u; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t before[FT_TT_QUEUE_STATUS_WORDS];
        uint32_t after[FT_TT_QUEUE_STATUS_WORDS];

        if ((group->resident_processor_mask & (1u << processor)) == 0u)
            continue;
        if (ft_tt_queue_read_status(noc, stuck, processor, before) != FT_TT_OK)
            continue;
        if (ft_tt_queue_read_status(noc, stuck, processor, after) != FT_TT_OK)
            continue;
        if (after[FT_TT_QUEUE_STATUS_HEARTBEAT] != before[FT_TT_QUEUE_STATUS_HEARTBEAT])
            return true;
    }
    return false;
}

/**
 * Name the processor a stalled core is stuck in, and the wait it is stuck in.
 *
 * "The group is stuck" is not a diagnosis. The one that is blocked is the one whose cursor
 * is LOWEST — the pipeline stalls behind its slowest stage — and reporting its state
 * distinguishes "the compute thread is in cb_wait_front because the reader is starved" from
 * "the reader is in a semaphore wait" without a second run.
 */
static void identify_stalled_processor(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                                       const ft_tt_queue *stuck, ft_tt_watchdog_report *report)
{
    uint32_t lowest_cursor = 0xFFFFFFFFu;
    unsigned processor;

    report->stalled_processor = FT_TT_PROC_COUNT;
    if (group->resident_processor_mask == 0u)
        return;

    for (processor = 0u; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t status[FT_TT_QUEUE_STATUS_WORDS];

        if ((group->resident_processor_mask & (1u << processor)) == 0u)
            continue;
        if (ft_tt_queue_read_status(noc, stuck, processor, status) != FT_TT_OK)
            continue;
        if (status[FT_TT_QUEUE_STATUS_STATE] == (uint32_t)FT_TT_QUEUE_STATE_EXIT)
            continue;
        if (status[FT_TT_QUEUE_STATUS_CURSOR] < lowest_cursor) {
            lowest_cursor = status[FT_TT_QUEUE_STATUS_CURSOR];
            report->stalled_processor = processor;
            report->stalled_processor_state = status[FT_TT_QUEUE_STATUS_STATE];
            report->stalled_processor_cursor = status[FT_TT_QUEUE_STATUS_CURSOR];
        }
    }
    if (report->stalled_processor < FT_TT_PROC_COUNT)
        ft_tt_log_error("watchdog:   the stalled processor is %s, in %s at descriptor %u",
                        ft_tt_processor_name(report->stalled_processor),
                        ft_tt_queue_state_string(report->stalled_processor_state),
                        report->stalled_processor_cursor);
}

int ft_tt_watchdog_wait(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                        uint32_t target_per_core, const ft_tt_watchdog_config *config,
                        ft_tt_watchdog_report *out_report)
{
    ft_tt_watchdog_report report;
    uint64_t started;
    uint64_t overall_deadline;
    uint64_t stall_deadline;
    uint32_t last_total = 0u;
    unsigned count;

    if (!noc || !noc->read || !group || !config)
        return FT_TT_EINVAL;
    /* Same refusal as ft_tt_launch_wait_done(): there is no untimed path here. */
    if (config->overall_timeout_ms == 0u || config->stall_timeout_ms == 0u)
        return FT_TT_EINVAL;

    count = ft_tt_launch_core_set_count(&group->cores);
    if (count == 0u || !group->queues)
        return FT_TT_EINVAL;

    memset(&report, 0, sizeof(report));
    /* Not 0: 0 is FT_TT_PROC_DM0, a real answer. "No processor identified" has to be its
     * own value or a caller reads BRISC out of a zeroed struct. */
    report.stalled_processor = FT_TT_PROC_COUNT;
    report.cores = count;
    report.completed_target = target_per_core * (uint32_t)count;

    started = monotonic_microseconds();
    overall_deadline = started + (uint64_t)config->overall_timeout_ms * 1000u;
    stall_deadline = started + (uint64_t)config->stall_timeout_ms * 1000u;

    for (;;) {
        uint32_t total = 0u;
        uint32_t short_slot = 0u;
        bool all_reached = false;
        uint64_t now;
        int rc = sample_group(noc, group, target_per_core, &total, &short_slot, &all_reached);

        if (rc != FT_TT_OK)
            return rc;

        report.completed_total = total;
        report.stalled_slot = short_slot;

        if (all_reached) {
            report.elapsed_ms = (uint32_t)((monotonic_microseconds() - started) / 1000u);
            if (out_report)
                *out_report = report;
            return FT_TT_OK;
        }

        now = monotonic_microseconds();
        if (total != last_total) {
            /* Progress. The stall clock is about "nothing finished", not "not finished". */
            last_total = total;
            stall_deadline = now + (uint64_t)config->stall_timeout_ms * 1000u;
        }

        if (now >= stall_deadline || now >= overall_deadline) {
            uint32_t heartbeat_before = 0u;
            uint32_t heartbeat_after = 0u;
            const ft_tt_queue *stuck = &group->queues[short_slot];

            report.elapsed_ms = (uint32_t)((now - started) / 1000u);

            /* ALIVE vs PROGRESSING. Two heartbeat reads a moment apart is the cheapest
             * way to tell a deadlocked kernel (recoverable by asking) from a core that
             * is not running our loop at all (a different problem entirely). */
            (void)ft_tt_queue_read_field(noc, stuck, FT_TT_QUEUE_FIELD_HEARTBEAT,
                                         &heartbeat_before);
            (void)ft_tt_queue_read_field(noc, stuck, FT_TT_QUEUE_FIELD_HEARTBEAT,
                                         &heartbeat_after);
            report.stalled_core_is_alive = heartbeat_after != heartbeat_before;
            /* For a TRIAD the control block's heartbeat is the READER's alone, and the
             * reader is usually the first processor to leave — so a core with four
             * processors still spinning reads NOT TICKING off that one word, which is the
             * opposite of the truth and points the operator at the mailbox instead of at
             * the ask. The core is ALIVE if ANY resident processor's heartbeat is moving. */
            if (group->resident_processor_mask != 0u)
                report.stalled_core_is_alive = any_processor_ticking(noc, group, stuck);
            (void)ft_tt_queue_read_field(noc, stuck, FT_TT_QUEUE_FIELD_STATE,
                                         &report.stalled_state);
            (void)ft_tt_queue_read_field(noc, stuck, FT_TT_QUEUE_FIELD_ERROR,
                                         &report.stalled_error);
            identify_stalled_processor(noc, group, stuck, &report);

            ft_tt_log_error("watchdog: group %u %s after %u ms — %u of %u descriptors completed; "
                            "core slot %u is short and is %s",
                            group->id, now >= stall_deadline ? "stalled" : "ran out of time",
                            report.elapsed_ms, total, report.completed_target, short_slot,
                            report.stalled_core_is_alive ? "ALIVE (blocked — ask it to stop)"
                                                         : "NOT TICKING (read the mailbox)");
            ft_tt_watchdog_dump(noc, group);

            if (out_report)
                *out_report = report;
            return FT_TT_EIO;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* The dump                                                                  */
/* ------------------------------------------------------------------------- */

void ft_tt_watchdog_dump(const ft_tt_noc_access *noc, const ft_tt_core_group *group)
{
    unsigned count;
    unsigned slot;

    if (!noc || !noc->read || !group)
        return;

    count = ft_tt_launch_core_set_count(&group->cores);
    ft_tt_log_error("watchdog: group %u dump, %u core(s), Tensix indices %u..%u", group->id, count,
                    group->first_tensix_index,
                    group->first_tensix_index + group->tensix_index_count - 1u);

    for (slot = 0u; slot < count; slot++) {
        const ft_tt_launch_core_placement *placement =
            ft_tt_launch_core_set_at(&group->cores, slot);
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        ft_tt_launch_state state;

        if (ft_tt_queue_read_control(noc, &group->queues[slot], fields) == FT_TT_OK) {
            ft_tt_log_error("  slot %u (%u, %u): state=%s head=%u tail=%u completed=%u token=0x%08x "
                            "last_opcode=%s error=%s idle_spins=%u heartbeat=%u stop=%u",
                            slot, (unsigned)placement->x, (unsigned)placement->y,
                            ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
                            fields[FT_TT_QUEUE_FIELD_HEAD], fields[FT_TT_QUEUE_FIELD_TAIL],
                            fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
                            fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN],
                            ft_tt_queue_opcode_string(fields[FT_TT_QUEUE_FIELD_LAST_OPCODE]),
                            ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]),
                            fields[FT_TT_QUEUE_FIELD_IDLE_SPINS],
                            fields[FT_TT_QUEUE_FIELD_HEARTBEAT],
                            fields[FT_TT_QUEUE_FIELD_STOP_REQUEST]);
        } else {
            ft_tt_log_error("  slot %u (%u, %u): the queue control block could not be read", slot,
                            (unsigned)placement->x, (unsigned)placement->y);
        }

        /* The PER-PROCESSOR half, for a triad. "The core is blocked" is not a diagnosis
         * when five processors share one queue: what is wanted is that TRISC0 is in
         * CB_WAIT_FRONT at descriptor 12 while BRISC is idle, which names both the
         * processor and the wait. Two heartbeat reads a moment apart per processor is what
         * separates blocked-but-alive from not-executing. */
        if (group->resident_processor_mask != 0u) {
            unsigned processor;

            for (processor = 0u; processor < FT_TT_PROC_COUNT; processor++) {
                uint32_t status[FT_TT_QUEUE_STATUS_WORDS];
                uint32_t again[FT_TT_QUEUE_STATUS_WORDS];

                if ((group->resident_processor_mask & (1u << processor)) == 0u)
                    continue;
                if (ft_tt_queue_read_status(noc, &group->queues[slot], processor, status) !=
                    FT_TT_OK)
                    continue;
                if (ft_tt_queue_read_status(noc, &group->queues[slot], processor, again) !=
                    FT_TT_OK)
                    again[FT_TT_QUEUE_STATUS_HEARTBEAT] = status[FT_TT_QUEUE_STATUS_HEARTBEAT];
                ft_tt_log_error("    %-6s state=%-16s cursor=%u opcode=%s error=%s heartbeat=%u %s",
                                ft_tt_processor_name(processor),
                                ft_tt_queue_state_string(status[FT_TT_QUEUE_STATUS_STATE]),
                                status[FT_TT_QUEUE_STATUS_CURSOR],
                                ft_tt_queue_opcode_string(status[FT_TT_QUEUE_STATUS_LAST_OPCODE]),
                                ft_tt_queue_kernel_error_string(status[FT_TT_QUEUE_STATUS_ERROR]),
                                status[FT_TT_QUEUE_STATUS_HEARTBEAT],
                                again[FT_TT_QUEUE_STATUS_HEARTBEAT] !=
                                        status[FT_TT_QUEUE_STATUS_HEARTBEAT]
                                    ? "ALIVE"
                                    : "NOT TICKING");
            }
        }

        /* The mailbox half. It is what says whether the firmware ever handed control to
         * the kernel, which the queue block cannot: an unlaunched core's control block
         * still reads exactly as the host wrote it. */
        if (ft_tt_launch_read_state(noc, placement->x, placement->y, &state) == FT_TT_OK)
            ft_tt_launch_dump_state(&state, placement->x, placement->y);
    }
}

/* ------------------------------------------------------------------------- */
/* Shutdown — by asking                                                      */
/* ------------------------------------------------------------------------- */

/**
 * Wait for a core's KERNEL to have EXITED (kernels retire, cores do not), which for a triad
 * means all five processors have left
 * their loops — not just the one that owns the control block's `state` word.
 *
 * `processor_mask` 0 keeps the BRISC-only behaviour, where the control block IS the whole
 * answer because there is one processor and it is that word's writer.
 */
static int wait_for_exit(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                         uint32_t processor_mask, uint32_t timeout_ms, uint32_t *out_pending)
{
    uint64_t deadline = monotonic_microseconds() + (uint64_t)timeout_ms * 1000u;

    for (;;) {
        uint32_t pending = 0u;
        int rc = ft_tt_queue_all_processors_exited(noc, queue, processor_mask, &pending);

        if (rc == FT_TT_OK) {
            if (out_pending)
                *out_pending = 0u;
            return FT_TT_OK;
        }
        if (rc != FT_TT_EIO)
            return rc;
        if (out_pending)
            *out_pending = pending;
        if (monotonic_microseconds() >= deadline) {
            /* A kernel that never started cannot be asked to stop, and saying so is more
             * useful than timing out on it. */
            uint32_t state = 0u;

            if (processor_mask == 0u &&
                ft_tt_queue_read_field(noc, queue, FT_TT_QUEUE_FIELD_STATE, &state) == FT_TT_OK &&
                state == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
                return FT_TT_ERROR;
            return FT_TT_EIO;
        }
    }
}

int ft_tt_watchdog_shutdown(const ft_tt_noc_access *noc, const ft_tt_core_group *group,
                            uint32_t timeout_ms, uint32_t *out_unstopped)
{
    unsigned count;
    unsigned slot;
    uint32_t unstopped = 0u;
    int first_error = FT_TT_OK;

    if (!noc || !group || timeout_ms == 0u)
        return FT_TT_EINVAL;

    count = ft_tt_launch_core_set_count(&group->cores);
    if (count == 0u || !group->queues)
        return FT_TT_EINVAL;

    /* Ask the WHOLE group first, then collect. Asking one core and waiting for it before
     * asking the next serialises the shutdown and, worse, lets a group deadlocked on a
     * cross-core semaphore keep the un-asked cores blocked. */
    for (slot = 0u; slot < count; slot++) {
        int rc = ft_tt_queue_request_stop(noc, &group->queues[slot]);
        if (rc != FT_TT_OK && first_error == FT_TT_OK)
            first_error = rc;
    }

    for (slot = 0u; slot < count; slot++) {
        uint32_t pending = 0u;
        int rc = wait_for_exit(noc, &group->queues[slot], group->resident_processor_mask, timeout_ms,
                               &pending);
        if (rc != FT_TT_OK) {
            const ft_tt_launch_core_placement *placement =
                ft_tt_launch_core_set_at(&group->cores, slot);
            unsigned processor;

            unstopped++;
            ft_tt_log_error("watchdog: group %u core (%u, %u) did not honour the stop request — "
                            "LEAVING IT ALONE. Asserting reset on a blocked kernel is what wedged "
                            "Tensix 0 (dev notes §1); dump the core and decide by hand",
                            group->id, (unsigned)placement->x, (unsigned)placement->y);
            /* WHICH processor, not just "the core". With five of them on one queue, "the
             * core did not stop" is not a diagnosis. */
            for (processor = 0u; processor < FT_TT_PROC_COUNT; processor++) {
                uint32_t status[FT_TT_QUEUE_STATUS_WORDS];

                if ((pending & (1u << processor)) == 0u)
                    continue;
                if (ft_tt_queue_read_status(noc, &group->queues[slot], processor, status) !=
                    FT_TT_OK)
                    continue;
                ft_tt_log_error("    %s is still in %s at descriptor %u (error=%s)",
                                ft_tt_processor_name(processor),
                                ft_tt_queue_state_string(status[FT_TT_QUEUE_STATUS_STATE]),
                                status[FT_TT_QUEUE_STATUS_CURSOR],
                                ft_tt_queue_kernel_error_string(status[FT_TT_QUEUE_STATUS_ERROR]));
            }
            if (first_error == FT_TT_OK)
                first_error = FT_TT_EIO;
        }
    }

    if (unstopped == 0u) {
        /* Now, and only now, the run message means something: the kernel already told us
         * it left its loop, so DONE is confirmation rather than evidence.
         *
         * This does NOT park. A parked core cannot be launched again without re-booting
         * its firmware, and "retire the resident kernel" is a different decision from
         * "stop using this core" — folding them together turns every shutdown into a
         * reboot, and it is a park that is dangerous when a kernel is still blocked. The
         * caller parks on its own exit path. */
        unsigned failed_index = 0u;
        (void)ft_tt_launch_wait_done_all(noc, &group->cores, timeout_ms, &failed_index);
    }

    if (out_unstopped)
        *out_unstopped = unstopped;
    return first_error;
}
