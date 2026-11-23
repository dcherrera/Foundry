/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_recover.c
 * @brief Implementation of the three core-level recovery rungs.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#include "ft_tt_core_recover.h"

#include <stdio.h>
#include <string.h>

#include "ft_tt_memmap.h"
#include "ft_tt_op_l1_layout.h"

/** Pages granted to a blocked consumer by rung 2's counter poke. Above every page count
 *  this backend configures, so one write satisfies any outstanding wait. */
#define POKE_PAGE_GRANT 64u

const char *ft_tt_core_recovery_rung_string(ft_tt_core_recovery_rung rung)
{
    switch (rung) {
    case FT_TT_RECOVERY_RUNG_DEASSERT_RISCS:  return "rung 0 — deassert the RISC-Vs";
    case FT_TT_RECOVERY_RUNG_RELOAD_FIRMWARE: return "rung 1 — reload firmware and boot";
    case FT_TT_RECOVERY_RUNG_POKE_WAIT:       return "rung 2 — poke the wait";
    }
    return "?";
}

/** The four sync bytes as one word, NCRISC first, for a compact before/after record. */
static uint32_t pack_subordinate_sync(const uint8_t sync[FT_TT_SUBORDINATE_MAP_SIZE])
{
    return (uint32_t)sync[0] | ((uint32_t)sync[1] << 8) | ((uint32_t)sync[2] << 16) |
           ((uint32_t)sync[3] << 24);
}

/** Fill the common half of a report and record the before-state. */
static void begin_report(const ft_tt_core_status *status, ft_tt_core_recovery_rung rung,
                         ft_tt_core_recovery_report *out)
{
    memset(out, 0, sizeof(*out));
    out->index = status->index;
    out->x = status->x;
    out->y = status->y;
    out->rung = rung;
    out->reset_register_before = status->state.reset_register;
    out->subordinate_sync_before = pack_subordinate_sync(status->state.subordinate_sync);
}

/** Re-read what the rung changed, so the report's "after" is measured, not assumed. */
static void end_report(const ft_tt_noc_access *noc, ft_tt_core_recovery_report *out)
{
    ft_tt_launch_state state;

    if (ft_tt_launch_read_state(noc, out->x, out->y, &state) != FT_TT_OK)
        return;
    out->reset_register_after = state.reset_register;
    out->subordinate_sync_after = pack_subordinate_sync(state.subordinate_sync);
}

int ft_tt_core_recover_deassert_riscs(const ft_tt_noc_access *noc,
                                      const ft_tt_core_status *status,
                                      ft_tt_core_recovery_report *out)
{
    if (!noc || !status || !out)
        return FT_TT_EINVAL;

    begin_report(status, FT_TT_RECOVERY_RUNG_DEASSERT_RISCS, out);
    /* ALL_RISCS and nothing else. The backend bits are not in this mask and must never be
     * written (ft_tt_launch.h). `staggered` is meaningless for a single core. */
    out->result = ft_tt_launch_deassert_reset(noc, out->x, out->y, FT_TT_SOFT_RESET_ALL_RISCS,
                                              false);
    end_report(noc, out);
    return FT_TT_OK;
}

int ft_tt_core_recover_reload_firmware(const ft_tt_noc_access *noc,
                                       const ft_tt_core_status *status,
                                       const ft_tt_firmware_set *firmware, bool force,
                                       ft_tt_core_recovery_report *out)
{
    if (!noc || !status || !firmware || !out)
        return FT_TT_EINVAL;

    begin_report(status, FT_TT_RECOVERY_RUNG_RELOAD_FIRMWARE, out);
    if (!force && status->verdict == FT_TT_CORE_VERDICT_KERNEL_RUNNING) {
        out->result = FT_TT_EPERM;
        return FT_TT_EPERM;
    }
    out->result = ft_tt_core_boot_firmware(noc, out->x, out->y, firmware, NULL);
    end_report(noc, out);
    return FT_TT_OK;
}

/** Poke 1 — ask a resident kernel to retire, if one left a control block behind. */
static int request_queue_stop(const ft_tt_noc_access *noc, const ft_tt_core_status *status,
                              ft_tt_core_recovery_report *out)
{
    uint32_t stop = 1u;
    int rc;

    if (!status->queue_present)
        return FT_TT_OK;
    rc = noc->write(noc->user, out->x, out->y,
                    FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_FIELD_STOP_REQUEST * 4u, &stop,
                    sizeof(stop));
    if (rc == FT_TT_OK)
        out->queue_stop_requested = true;
    return rc;
}

/** Poke 2 — tell a BRISC in `tensix_sync()` that its subordinates are done. */
static int release_subordinate_sync(const ft_tt_noc_access *noc, ft_tt_core_recovery_report *out)
{
    uint32_t all_done = FT_TT_RUN_SYNC_MSG_ALL_SUBORDINATES_DONE;

    return noc->write(noc->user, out->x, out->y, FT_TT_MB_SUBORDINATE_SYNC_ADDR, &all_done,
                      sizeof(all_done));
}

/** Poke 3 — grant pages on every circular buffer, releasing any blocked consumer. */
static int grant_circular_buffer_pages(const ft_tt_noc_access *noc,
                                       ft_tt_core_recovery_report *out)
{
    unsigned cb_index;

    for (cb_index = 0; cb_index < FT_TT_NUM_CIRCULAR_BUFFERS; cb_index++) {
        uint32_t acked = 0;
        uint32_t granted;
        int rc = noc->read(noc->user, out->x, out->y, FT_TT_CB_TILES_ACKED_ADDR(cb_index), &acked,
                           sizeof(acked));

        if (rc != FT_TT_OK)
            return rc;
        /* The wait is `received - acked >= n` in 16-bit arithmetic, so the grant is
         * relative to what the consumer has already freed. */
        granted = (acked + POKE_PAGE_GRANT) & 0xFFFFu;
        rc = noc->write(noc->user, out->x, out->y, FT_TT_CB_TILES_RECEIVED_ADDR(cb_index),
                        &granted, sizeof(granted));
        if (rc != FT_TT_OK)
            return rc;
        out->circular_buffers_poked++;
    }
    return FT_TT_OK;
}

int ft_tt_core_recover_poke_wait(const ft_tt_noc_access *noc, const ft_tt_core_status *status,
                                 ft_tt_core_recovery_report *out)
{
    int rc;

    if (!noc || !status || !out)
        return FT_TT_EINVAL;

    begin_report(status, FT_TT_RECOVERY_RUNG_POKE_WAIT, out);
    rc = request_queue_stop(noc, status, out);
    if (rc == FT_TT_OK)
        rc = release_subordinate_sync(noc, out);
    if (rc == FT_TT_OK)
        rc = grant_circular_buffer_pages(noc, out);
    out->result = rc;
    end_report(noc, out);
    return FT_TT_OK;
}

void ft_tt_core_recovery_print(const ft_tt_core_recovery_report *report)
{
    if (!report)
        return;

    printf("Tensix %u (%u, %u) — %s\n", report->index, report->x, report->y,
           ft_tt_core_recovery_rung_string(report->rung));
    printf("  writes returned rc=%d\n", report->result);
    if (report->queue_stop_requested)
        printf("  resident-queue stop flag set\n");
    if (report->circular_buffers_poked != 0u)
        printf("  granted %u pages on each of %u circular buffers\n", POKE_PAGE_GRANT,
               report->circular_buffers_poked);
    printf("  soft reset      0x%08x -> 0x%08x\n", report->reset_register_before,
           report->reset_register_after);
    printf("  subordinate sync 0x%08x -> 0x%08x\n", report->subordinate_sync_before,
           report->subordinate_sync_after);
}
