/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_relaunch_gate.c — the re-launch verdict, OFF HARDWARE, one mutation per ground.
 *
 * This is the file that makes stage 3c of tests/test_multicast_bench.c worth running.
 * That stage answers "does a wide multicast damage a core, or does our park + re-boot
 * teardown?", and it answers it with a NEGATIVE — "the second launch was fine". A negative
 * is only evidence if the instrument could have said otherwise, and this project has
 * shipped four gates that could not. So every ground the gate can fail
 * on is driven here, on the Mac, with no card:
 *
 *  - ONE healthy record, which must come back CLEAN. If it does not, every red below is
 *    meaningless because the gate refuses everything.
 *  - ONE mutation per ground, each of which must come back as ITS OWN outcome and not
 *    merely "not clean". A check with several sufficient grounds fails on whichever is
 *    cheapest and then passes for the right reason and fails for the wrong one, which no
 *    green run can distinguish.
 *  - The THREE VACUITY grounds are driven on records that are otherwise PERFECT. That is
 *    the point of them: they must beat a green, not merely accompany a red.
 *
 * Deleting any single `if` from ft_tt_relaunch_gate.c makes exactly one check here go red,
 * which is the property being bought.
 *
 * Build and run:
 *   make -f Makefile.multicast check
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_relaunch_gate.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

/** A probe that went perfectly, on a core that needed no re-boot. Every mutation below
 *  starts from this and changes exactly one field, so what a check proves is the field it
 *  changed and nothing else. */
static ft_tt_relaunch_evidence healthy_record(void)
{
    ft_tt_relaunch_evidence evidence;

    memset(&evidence, 0, sizeof(evidence));
    evidence.launch_rc = 0;
    evidence.status_poisoned_before = true;
    evidence.destination_poisoned_before = true;
    evidence.every_status_magic_present = true;
    evidence.kernel_error = 0u;
    evidence.rounds_complete = true;
    evidence.payload_matched_demanded = true;
    evidence.payload_rejected_other = true;
    evidence.previous_block_index = FT_TT_RELAUNCH_BLOCK_A;
    evidence.demanded_block_index = FT_TT_RELAUNCH_BLOCK_B;
    return evidence;
}

static void check_outcome(const char *what, const ft_tt_relaunch_evidence *evidence,
                          ft_tt_relaunch_outcome wanted)
{
    ft_tt_relaunch_outcome got = ft_tt_relaunch_gate_judge(evidence);

    ft_tt_test_check_fmt(what, got == wanted, "got %d (%s), wanted %d (%s)", (int)got,
                         ft_tt_relaunch_outcome_text(got), (int)wanted,
                         ft_tt_relaunch_outcome_text(wanted));
}

/* ------------------------------------------------------------------------- */

static void check_a_healthy_probe_is_clean(void)
{
    ft_tt_relaunch_evidence evidence = healthy_record();

    ft_tt_test_step("a probe that went perfectly is CLEAN — without this every red below "
                    "is a gate that refuses everything");

    check_outcome("a healthy record is CLEAN", &evidence, FT_TT_RELAUNCH_CLEAN);
    ft_tt_test_check("and CLEAN is the only outcome that reads as clean",
                     ft_tt_relaunch_outcome_is_clean(FT_TT_RELAUNCH_CLEAN) &&
                         !ft_tt_relaunch_outcome_is_clean(FT_TT_RELAUNCH_PAYLOAD_WRONG) &&
                         !ft_tt_relaunch_outcome_is_clean(FT_TT_RELAUNCH_VACUOUS_SAME_BLOCK),
                     "ft_tt_relaunch_outcome_is_clean over three outcomes");
}

/**
 * The three vacuity grounds, each on an OTHERWISE PERFECT record.
 *
 * This is the half of the suite that protects the multicast answer. A stage 3c that came
 * back green because it never exercised anything would be read as "the broadcast is
 * innocent" and would send the cooperative-matmul design down a path chosen by a bug.
 */
static void check_a_probe_that_could_not_have_failed_beats_a_green(void)
{
    ft_tt_relaunch_evidence evidence;

    ft_tt_test_step("a probe that COULD NOT have failed is refused, not passed");

    evidence = healthy_record();
    evidence.status_poisoned_before = false;
    check_outcome("an unpoisoned status block is vacuous, not clean", &evidence,
                  FT_TT_RELAUNCH_VACUOUS_STATUS_NOT_POISONED);

    evidence = healthy_record();
    evidence.destination_poisoned_before = false;
    check_outcome("an unpoisoned destination is vacuous, not clean", &evidence,
                  FT_TT_RELAUNCH_VACUOUS_DESTINATION_NOT_POISONED);

    /* The one that is easy to produce by accident: pick a round count of the wrong parity
     * and the probe demands the block the preceding multicast already left, so the
     * comparison passes with nothing moving. */
    evidence = healthy_record();
    evidence.demanded_block_index = evidence.previous_block_index;
    check_outcome("demanding the block already in L1 is vacuous, not clean", &evidence,
                  FT_TT_RELAUNCH_VACUOUS_SAME_BLOCK);

    evidence = healthy_record();
    evidence.previous_block_index = FT_TT_RELAUNCH_BLOCK_B;
    evidence.demanded_block_index = FT_TT_RELAUNCH_BLOCK_A;
    check_outcome("and the same record with the blocks the other way round is CLEAN",
                  &evidence, FT_TT_RELAUNCH_CLEAN);

    evidence = healthy_record();
    evidence.demanded_block_index = 7u;
    check_outcome("a block index outside {A,B} is a malformed record", &evidence,
                  FT_TT_RELAUNCH_INVALID_EVIDENCE);

    check_outcome("and a NULL record is malformed too", NULL, FT_TT_RELAUNCH_INVALID_EVIDENCE);
}

/** The five ways the CORE can fail, one mutation each. */
static void check_every_failure_ground_names_itself(void)
{
    ft_tt_relaunch_evidence evidence;

    ft_tt_test_step("each way a core can fail reports ITS OWN ground");

    evidence = healthy_record();
    evidence.launch_rc = -5;
    check_outcome("a launch that failed or timed out", &evidence, FT_TT_RELAUNCH_LAUNCH_FAILED);

    evidence = healthy_record();
    evidence.every_status_magic_present = false;
    check_outcome("a status block still holding the host poison", &evidence,
                  FT_TT_RELAUNCH_NEVER_RAN);

    evidence = healthy_record();
    evidence.kernel_error = 0xFFFFFFFFu;
    check_outcome("a kernel-side refusal", &evidence, FT_TT_RELAUNCH_KERNEL_ERROR);

    evidence = healthy_record();
    evidence.rounds_complete = false;
    check_outcome("a core that stopped early", &evidence, FT_TT_RELAUNCH_SHORT_RUN);

    evidence = healthy_record();
    evidence.payload_matched_demanded = false;
    check_outcome("a destination holding the wrong bytes", &evidence,
                  FT_TT_RELAUNCH_PAYLOAD_WRONG);

    evidence = healthy_record();
    evidence.payload_rejected_other = false;
    check_outcome("a destination still holding the previous block", &evidence,
                  FT_TT_RELAUNCH_PAYLOAD_STALE);
}

/**
 * The ORDER is part of the contract: vacuity outranks a real failure too.
 *
 * Not a stylistic point. A probe that was never poisoned and whose core also failed must
 * be reported as vacuous, because the failure cannot be attributed — an unpoisoned
 * destination holding the wrong bytes says nothing about this launch.
 */
static void check_vacuity_outranks_a_real_failure(void)
{
    ft_tt_relaunch_evidence evidence = healthy_record();

    ft_tt_test_step("vacuity outranks a real failure, so nothing is attributed to a probe "
                    "that proves nothing");

    evidence.status_poisoned_before = false;
    evidence.launch_rc = -5;
    evidence.payload_matched_demanded = false;
    check_outcome("an unpoisoned probe whose core ALSO failed is reported vacuous",
                  &evidence, FT_TT_RELAUNCH_VACUOUS_STATUS_NOT_POISONED);
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    ft_tt_test_step("the re-launch verdict — off hardware, one mutation per ground");

    check_a_healthy_probe_is_clean();
    check_a_probe_that_could_not_have_failed_beats_a_green();
    check_every_failure_ground_names_itself();
    check_vacuity_outranks_a_real_failure();

    return ft_tt_test_finish();
}
