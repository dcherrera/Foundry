/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_status.c
 * @brief Implementation of the read-only per-Tensix status read and its verdict.
 *
 * MOVED OUT OF tt/tools/ 2026-08-24 and it is a LIBRARY file now, because a second consumer
 * appeared: Pantheon's arming path has to ask "is this core ready to have firmware booted
 * onto it" before it boots, and that is the question this verdict already answers. Copying
 * the classification into the serve would have produced a SECOND, subtly different reading
 * of the same registers — see ft_tt_core_readiness.h for what the arming path adds on top.
 * The three printers stayed behind in tools/ (ft_tt_core_status_print.c): a library that
 * reads a register has no business writing to stdout.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#include "ft_tt_core_status.h"

#include <string.h>

#include "ft_tt_memmap.h"
#include "ft_tt_op_l1_layout.h"

bool ft_tt_core_status_processor_is_held(uint32_t reset_register, unsigned processor)
{
    return (reset_register & ft_tt_processor_reset_bit(processor)) != 0u;
}

/** Local spelling of the exported predicate, so the classifier below reads as it always did. */
#define processor_is_held ft_tt_core_status_processor_is_held

const char *ft_tt_core_verdict_string(ft_tt_core_verdict verdict)
{
    switch (verdict) {
    case FT_TT_CORE_VERDICT_NO_ANSWER:      return "no answer";
    case FT_TT_CORE_VERDICT_HELD:           return "held in reset";
    case FT_TT_CORE_VERDICT_FIRMWARE_IDLE:  return "firmware idle";
    case FT_TT_CORE_VERDICT_BOOT_INCOMPLETE: return "boot handshake never finished";
    case FT_TT_CORE_VERDICT_KERNEL_RUNNING: return "kernel running";
    case FT_TT_CORE_VERDICT_SYNC_BLOCKED:   return "BRISC blocked in tensix_sync";
    case FT_TT_CORE_VERDICT_MIXED:          return "mixed reset state";
    }
    return "?";
}

ft_tt_core_verdict ft_tt_core_status_classify(const ft_tt_core_status *status)
{
    const ft_tt_launch_state *state = &status->state;
    unsigned held_count = 0;
    unsigned subordinate;
    unsigned processor;
    bool brisc_held;
    bool subordinates_all_held = true;
    bool subordinate_sync_pending = false;

    if (status->read_result != FT_TT_OK)
        return FT_TT_CORE_VERDICT_NO_ANSWER;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++)
        if (processor_is_held(state->reset_register, processor))
            held_count++;
    if (held_count == FT_TT_PROC_COUNT)
        return FT_TT_CORE_VERDICT_HELD;

    brisc_held = processor_is_held(state->reset_register, FT_TT_PROC_DM0);
    for (processor = FT_TT_PROC_DM1; processor < FT_TT_PROC_COUNT; processor++)
        if (!processor_is_held(state->reset_register, processor))
            subordinates_all_held = false;
    /* subordinate_sync is [NCRISC, TRISC0, TRISC1, TRISC2];
     * 0x00 = done, 0x40 = init, 0x80 = go. Anything not DONE is a subordinate BRISC is
     * still waiting on. */
    for (subordinate = 0; subordinate < FT_TT_SUBORDINATE_MAP_SIZE; subordinate++)
        if (state->subordinate_sync[subordinate] != FT_TT_RUN_SYNC_MSG_DONE)
            subordinate_sync_pending = true;

    if (!brisc_held && subordinates_all_held && subordinate_sync_pending)
        return FT_TT_CORE_VERDICT_SYNC_BLOCKED;
    if (state->go_messages[0].signal == FT_TT_RUN_MSG_GO)
        return FT_TT_CORE_VERDICT_KERNEL_RUNNING;
    /* RUN_MSG_INIT is what the host primes the handshake with; BRISC overwrites it with
     * DONE once it and its four subordinates have initialised. Still INIT
     * with BRISC out of reset means that never happened. */
    if (!brisc_held && state->go_messages[0].signal == FT_TT_RUN_MSG_INIT)
        return FT_TT_CORE_VERDICT_BOOT_INCOMPLETE;
    if (!brisc_held && held_count == 0)
        return FT_TT_CORE_VERDICT_FIRMWARE_IDLE;
    return FT_TT_CORE_VERDICT_MIXED;
}

/** Read the resident-queue control block, if one of ours is there. */
static void read_queue_control(const ft_tt_noc_access *noc, ft_tt_core_status *status)
{
    uint32_t again[FT_TT_QUEUE_FIELD_COUNT];

    if (noc->read(noc->user, status->x, status->y, FT_TT_OP_QUEUE_BASE, status->queue_control,
                  FT_TT_QUEUE_CONTROL_BYTES) != FT_TT_OK)
        return;
    if (status->queue_control[FT_TT_QUEUE_FIELD_MAGIC] != FT_TT_QUEUE_MAGIC)
        return;
    status->queue_present = true;

    /* Two samples: a heartbeat that moves is the one signal here that a processor is
     * still executing rather than merely leaving plausible words behind. */
    if (noc->read(noc->user, status->x, status->y, FT_TT_OP_QUEUE_BASE, again,
                  FT_TT_QUEUE_CONTROL_BYTES) != FT_TT_OK)
        return;
    status->queue_heartbeat_ticking =
        again[FT_TT_QUEUE_FIELD_HEARTBEAT] != status->queue_control[FT_TT_QUEUE_FIELD_HEARTBEAT];
}

int ft_tt_core_status_read(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                           uint32_t index, ft_tt_core_status *out)
{
    ft_tt_noc_coord core;
    int rc;

    if (!noc || !grid || !out)
        return FT_TT_EINVAL;

    memset(out, 0, sizeof(*out));
    out->index = index;

    /* The coordinate comes from live telemetry every time. A guessed endpoint does not
     * fail a read, it hangs the chip (spec §7.3). */
    rc = ft_tt_coord_tensix_by_index(grid, index, FT_TT_COORD_TRANSLATED, &core);
    if (rc != FT_TT_OK)
        return rc;
    out->x = (uint16_t)core.x;
    out->y = (uint16_t)core.y;

    out->read_result = ft_tt_launch_read_state(noc, out->x, out->y, &out->state);
    if (out->read_result == FT_TT_OK)
        read_queue_control(noc, out);
    out->verdict = ft_tt_core_status_classify(out);
    return FT_TT_OK;
}
