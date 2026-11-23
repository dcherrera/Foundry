/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_relaunch_gate.c
 * @brief Implementation of the re-launch probe verdict. Pure arithmetic over a record of
 *   observations; it opens nothing and reads nothing.
 *
 * @status Complete. Gated off-hardware by tests/test_relaunch_gate.c, one mutation per
 *   ground.
 * @issues None
 * @todo None
 */

#include "ft_tt_relaunch_gate.h"

static bool block_index_is_legal(uint32_t index)
{
    return index == FT_TT_RELAUNCH_BLOCK_A || index == FT_TT_RELAUNCH_BLOCK_B;
}

ft_tt_relaunch_outcome ft_tt_relaunch_gate_judge(const ft_tt_relaunch_evidence *evidence)
{
    if (evidence == NULL)
        return FT_TT_RELAUNCH_INVALID_EVIDENCE;

    /* Vacuity first, deliberately. A probe that could not have failed must never be
     * reported as a pass, whatever else the record says — the experiment this feeds is
     * decided by its NEGATIVE result. */
    if (!evidence->status_poisoned_before)
        return FT_TT_RELAUNCH_VACUOUS_STATUS_NOT_POISONED;
    if (!evidence->destination_poisoned_before)
        return FT_TT_RELAUNCH_VACUOUS_DESTINATION_NOT_POISONED;
    if (!block_index_is_legal(evidence->previous_block_index) ||
        !block_index_is_legal(evidence->demanded_block_index))
        return FT_TT_RELAUNCH_INVALID_EVIDENCE;
    if (evidence->previous_block_index == evidence->demanded_block_index)
        return FT_TT_RELAUNCH_VACUOUS_SAME_BLOCK;

    if (evidence->launch_rc != 0)
        return FT_TT_RELAUNCH_LAUNCH_FAILED;
    if (!evidence->every_status_magic_present)
        return FT_TT_RELAUNCH_NEVER_RAN;
    if (evidence->kernel_error != 0u)
        return FT_TT_RELAUNCH_KERNEL_ERROR;
    if (!evidence->rounds_complete)
        return FT_TT_RELAUNCH_SHORT_RUN;
    if (!evidence->payload_matched_demanded)
        return FT_TT_RELAUNCH_PAYLOAD_WRONG;
    if (!evidence->payload_rejected_other)
        return FT_TT_RELAUNCH_PAYLOAD_STALE;
    return FT_TT_RELAUNCH_CLEAN;
}

bool ft_tt_relaunch_outcome_is_clean(ft_tt_relaunch_outcome outcome)
{
    return outcome == FT_TT_RELAUNCH_CLEAN;
}

const char *ft_tt_relaunch_outcome_text(ft_tt_relaunch_outcome outcome)
{
    switch (outcome) {
    case FT_TT_RELAUNCH_CLEAN:
        return "the still-booted core accepted work and the payload tracked the source";
    case FT_TT_RELAUNCH_VACUOUS_STATUS_NOT_POISONED:
        return "VACUOUS: the status block was not poisoned, so \"never ran\" reads as DONE";
    case FT_TT_RELAUNCH_VACUOUS_DESTINATION_NOT_POISONED:
        return "VACUOUS: the destination was not poisoned, so a match may be stale L1";
    case FT_TT_RELAUNCH_VACUOUS_SAME_BLOCK:
        return "VACUOUS: the probe demanded the block already sitting in the destination";
    case FT_TT_RELAUNCH_INVALID_EVIDENCE:
        return "the record is malformed; nothing may be concluded from it";
    case FT_TT_RELAUNCH_LAUNCH_FAILED:
        return "the re-launch failed or timed out on a core that needed no re-boot";
    case FT_TT_RELAUNCH_NEVER_RAN:
        return "a status block still held the host poison: the kernel never ran";
    case FT_TT_RELAUNCH_KERNEL_ERROR:
        return "the kernel reported an error at the point of issue";
    case FT_TT_RELAUNCH_SHORT_RUN:
        return "a core stopped before its last round";
    case FT_TT_RELAUNCH_PAYLOAD_WRONG:
        return "a destination does not hold the demanded block";
    case FT_TT_RELAUNCH_PAYLOAD_STALE:
        return "a destination still holds the previous block: nothing landed";
    }
    return "unknown outcome";
}
