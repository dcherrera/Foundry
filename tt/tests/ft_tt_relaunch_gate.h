/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_relaunch_gate.h
 * @brief The verdict for a RE-LAUNCH probe: did a still-booted core accept fresh work,
 *   and — just as important — could the probe have said no?
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A SEPARATE MODULE AND NOT AN `if` INSIDE THE BENCH
 * ---------------------------------------------------------------------------
 * The experiment this serves (tests/test_multicast_bench.c stage 3c) exists to decide
 * whether a wide multicast damages a core, or whether our PARK + RE-BOOT teardown does.
 * Its whole value is in a NEGATIVE result: "the second launch was fine" only means
 * something if "the second launch was not fine" was reachable and would have been
 * reported. A decision that lives inline in a program that needs silicon cannot be
 * proven to go red without silicon, and this project has already shipped four gates that
 * turned out to measure nothing.
 *
 * So the decision is pure: no device, no I/O, no globals. tests/test_relaunch_gate.c
 * drives it on the Mac with one healthy record and one mutation per ground, and every
 * mutation must produce ITS OWN outcome — not merely "not clean". That is the DEV_NOTES
 * §3 rule about a check with several sufficient grounds: when a check can fail for two
 * reasons it must say which, or a green run cannot distinguish passing for the right
 * reason from failing for the wrong one.
 *
 * ---------------------------------------------------------------------------
 * THE VACUITY GROUNDS COME FIRST, AND THEY REFUSE A GREEN
 * ---------------------------------------------------------------------------
 * Three of the outcomes below are not failures of the CORE, they are failures of the
 * PROBE, and the gate reports them in preference to CLEAN even when every other field
 * says the run was perfect:
 *
 *   - the status block was not poisoned first, so "never ran" and "ran" are the same
 *     bit pattern (RUN_MSG_DONE is 0 — DEV_NOTES.md §3);
 *   - the destination was not poisoned first, so a match may be last run's bytes;
 *   - the probe demanded the SAME block the preceding run already left in L1, so the
 *     payload comparison would pass without a single byte moving.
 *
 * A probe that trips one of those has not exercised the core, and reporting it as CLEAN
 * would put a fabricated negative into the multicast decision. This is the one part of
 * the gate that has to be got right, because it is the part that protects the answer.
 *
 * @status Complete and off-hardware. Touches no device.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_RELAUNCH_GATE_H
#define FT_TT_RELAUNCH_GATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/** The two DRAM blocks the bench alternates between, by round parity. Named here because
 *  the gate compares "what the previous run left" against "what this probe demands" and a
 *  bare 0/1 at a call site is the kind of thing that gets swapped. */
#define FT_TT_RELAUNCH_BLOCK_A 0u
#define FT_TT_RELAUNCH_BLOCK_B 1u

/**
 * What one re-launch probe observed, as recorded by the caller.
 *
 * Every field is an OBSERVATION, never a conclusion — the gate does the concluding. A
 * caller that pre-judges (say, by only filling this in when things went well) defeats the
 * whole arrangement.
 */
typedef struct ft_tt_relaunch_evidence {
    /** Host-side result of writing the launch message, sending GO and waiting for DONE.
     *  0 is FT_TT_OK; anything else is a timeout or an I/O failure. */
    int launch_rc;
    /** Was every participating core's status block written with the host poison BEFORE
     *  the launch? Without it "the kernel never ran" reads as zero, and so does a
     *  perfectly ordinary DONE. */
    bool status_poisoned_before;
    /** Was every destination L1 region poisoned BEFORE the launch? */
    bool destination_poisoned_before;
    /** Did every core's status block come back carrying the kernel's magic word? */
    bool every_status_magic_present;
    /** First non-zero kernel-side error code across the participating cores; 0 is none. */
    uint32_t kernel_error;
    /** Did every core report exactly `iterations + 1` rounds? */
    bool rounds_complete;
    /** Does every destination hold the block this probe demanded? */
    bool payload_matched_demanded;
    /** And does every destination NOT hold the other block? */
    bool payload_rejected_other;
    /** Which block the run BEFORE this probe left in the destination. */
    uint32_t previous_block_index;
    /** Which block this probe's round count demands. Must differ from the above, or the
     *  comparison proves nothing. */
    uint32_t demanded_block_index;
} ft_tt_relaunch_evidence;

/**
 * The verdict. Ordered as the gate evaluates it: the three vacuity grounds, then the
 * invalid-record ground, then the five ways a core can actually fail.
 */
typedef enum ft_tt_relaunch_outcome {
    FT_TT_RELAUNCH_CLEAN = 0,
    /** The probe could not have failed — the status block was not poisoned first. */
    FT_TT_RELAUNCH_VACUOUS_STATUS_NOT_POISONED,
    /** The probe could not have failed — the destination was not poisoned first. */
    FT_TT_RELAUNCH_VACUOUS_DESTINATION_NOT_POISONED,
    /** The probe demanded the block that was already there. */
    FT_TT_RELAUNCH_VACUOUS_SAME_BLOCK,
    /** A block index outside {A, B}: the record itself is wrong, so nothing may be
     *  concluded from it in either direction. */
    FT_TT_RELAUNCH_INVALID_EVIDENCE,
    /** The launch itself failed or timed out. */
    FT_TT_RELAUNCH_LAUNCH_FAILED,
    /** A status block still held the host poison: the kernel never ran. */
    FT_TT_RELAUNCH_NEVER_RAN,
    /** The kernel refused or errored at the point of issue. */
    FT_TT_RELAUNCH_KERNEL_ERROR,
    /** A core stopped early. */
    FT_TT_RELAUNCH_SHORT_RUN,
    /** A destination does not hold the demanded block. */
    FT_TT_RELAUNCH_PAYLOAD_WRONG,
    /** A destination still holds the previous block: nothing landed this time. */
    FT_TT_RELAUNCH_PAYLOAD_STALE
} ft_tt_relaunch_outcome;

/**
 * Judge one probe.
 *
 * @param evidence  what was observed; never NULL
 * @return the FIRST ground that applies, in the order the enum declares — vacuity before
 *         anything else, so a probe that could not have failed is never reported as a
 *         pass. FT_TT_RELAUNCH_CLEAN only when no ground applies.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
ft_tt_relaunch_outcome ft_tt_relaunch_gate_judge(const ft_tt_relaunch_evidence *evidence);

/** Whether the outcome says the core accepted the work. Only CLEAN does. */
bool ft_tt_relaunch_outcome_is_clean(ft_tt_relaunch_outcome outcome);

/**
 * A short, stable sentence for the outcome — printed into the transcript a human reads
 * afterwards. Never NULL, including for a value outside the enum.
 */
const char *ft_tt_relaunch_outcome_text(ft_tt_relaunch_outcome outcome);

#endif /* FT_TT_RELAUNCH_GATE_H */
