/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_core_readiness.c
 * @brief Off-hardware unit test for the may-I-boot-here classifier
 *   (ft_tt_core_readiness.{h,c}) and for the verdict it rests on
 *   (ft_tt_core_status_classify()).
 *
 * TOUCHES NO DEVICE. Every case is a pure function of a hand-built
 * `ft_tt_core_status` and five hand-built liveness samples, so all of it runs on
 * a machine with no card in it. That is the point twice over: the code under
 * test decides whether firmware gets booted onto a Tensix, and getting that
 * wrong in the permissive direction costs a whole-card reset while getting it
 * wrong in the strict direction drops the whole model to the ~26 s/position CPU
 * forward. Neither is a thing to find out on silicon.
 *
 * ---------------------------------------------------------------------------
 * THE TWO ARMS THAT MATTER, AND THEY PULL IN OPPOSITE DIRECTIONS
 * ---------------------------------------------------------------------------
 *   RED  — the leftover: a reader that already exited, a compute thread whose
 *          heartbeat is still moving. The control block's own heartbeat is
 *          FROZEN in this case, which is exactly why `ft_tt_core_status`'s
 *          reading alone is not enough and this module samples the per-processor
 *          slots as well.
 *   GREEN — the healthy core, in each of the shapes a clean exit really leaves:
 *          cold and held, firmware idle with no queue at all, and firmware idle
 *          with a retired queue whose slots still hold EXIT and whose heartbeats
 *          have stopped. A guard that refuses ANY of these is worse than the bug.
 *
 * The third group is the stale-word group, and it is the false-positive trap
 * this classifier is specifically built around: a memory-family residency
 * (DM0|DM1) leaves its three TRISC slots holding whatever the previous program
 * wrote, because they never ran and never wrote EXIT. Those cases MUST come back
 * ready, and they are checked.
 *
 * @status New 2026-08-24.
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_core_readiness.h"
#include "ft_tt_core_status.h"
#include "ft_tt_launch.h"
#include "ft_tt_mailbox.h"
#include "ft_tt_memmap.h"
#include "ft_tt_queue.h"

static unsigned checks_run;
static unsigned checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

/* ------------------------------------------------------------------------- */
/* Building a reading by hand                                                 */
/* ------------------------------------------------------------------------- */

/** Every RISC-V held in reset — the cold/parked register value. */
static uint32_t all_held(void)
{
    uint32_t bits = 0u;
    unsigned processor;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++)
        bits |= ft_tt_processor_reset_bit(processor);
    return bits;
}

/**
 * A core that answers, with every subordinate synced DONE, a ready instruction
 * FIFO, and every RISC-V out of reset. `go` picks the go signal, which is what
 * separates FIRMWARE_IDLE from KERNEL_RUNNING.
 */
static void status_running(ft_tt_core_status *out, uint8_t go)
{
    memset(out, 0, sizeof(*out));
    out->index = 90u;
    out->x = 5u;
    out->y = 7u;
    out->read_result = FT_TT_OK;
    out->state.reset_register = 0u;
    out->state.instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_READY;
    out->state.go_messages[0].signal = go;
    /* subordinate_sync is already all-zero from the memset, and DONE is 0x00. */
    out->verdict = ft_tt_core_status_classify(out);
}

/** The cold core: everything held in reset. */
static void status_held(ft_tt_core_status *out)
{
    memset(out, 0, sizeof(*out));
    out->index = 90u;
    out->read_result = FT_TT_OK;
    out->state.reset_register = all_held();
    out->state.instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_READY;
    out->state.go_messages[0].signal = FT_TT_RUN_MSG_DONE;
    out->verdict = ft_tt_core_status_classify(out);
}

/** Mark a queue control block as present, so the slots are consulted at all. */
static void with_queue(ft_tt_core_status *status, bool control_heartbeat_ticking)
{
    status->queue_present = true;
    status->queue_control[FT_TT_QUEUE_FIELD_MAGIC] = FT_TT_QUEUE_MAGIC;
    status->queue_heartbeat_ticking = control_heartbeat_ticking;
}

/** Every slot read, holding `state`, none of them ticking. */
static void slots_quiet(ft_tt_core_processor_liveness *slots, uint32_t state)
{
    unsigned processor;

    memset(slots, 0, FT_TT_PROC_COUNT * sizeof(*slots));
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        slots[processor].slot_read = true;
        slots[processor].state = state;
        slots[processor].heartbeat = 4242u;
    }
}

/* ------------------------------------------------------------------------- */
/* The verdict itself — the backbone the readiness rests on                   */
/* ------------------------------------------------------------------------- */

static void verdict_cases(void)
{
    ft_tt_core_status status;

    printf("ft_tt_core_status_classify — the verdicts readiness rests on\n");

    status_held(&status);
    check(status.verdict == FT_TT_CORE_VERDICT_HELD, "all five in reset is HELD");

    status_running(&status, FT_TT_RUN_MSG_DONE);
    check(status.verdict == FT_TT_CORE_VERDICT_FIRMWARE_IDLE,
          "out of reset, synced, go DONE is FIRMWARE_IDLE");

    status_running(&status, FT_TT_RUN_MSG_GO);
    check(status.verdict == FT_TT_CORE_VERDICT_KERNEL_RUNNING, "go GO is KERNEL_RUNNING");

    status_running(&status, FT_TT_RUN_MSG_INIT);
    check(status.verdict == FT_TT_CORE_VERDICT_BOOT_INCOMPLETE,
          "go still INIT with BRISC running is BOOT_INCOMPLETE");

    /* The core-60 signature: BRISC out of reset, every subordinate held, and a
     * sync byte that is not DONE. BRISC is inside tensix_sync() waiting on
     * processors that can never answer. */
    memset(&status, 0, sizeof(status));
    status.read_result = FT_TT_OK;
    status.state.instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_READY;
    status.state.reset_register = all_held() & ~ft_tt_processor_reset_bit(FT_TT_PROC_DM0);
    status.state.subordinate_sync[1] = FT_TT_RUN_MSG_GO;
    status.verdict = ft_tt_core_status_classify(&status);
    check(status.verdict == FT_TT_CORE_VERDICT_SYNC_BLOCKED,
          "BRISC up, subordinates held, sync pending is SYNC_BLOCKED");

    memset(&status, 0, sizeof(status));
    status.read_result = FT_TT_EIO;
    status.verdict = ft_tt_core_status_classify(&status);
    check(status.verdict == FT_TT_CORE_VERDICT_NO_ANSWER, "a failed mailbox read is NO_ANSWER");
}

/* ------------------------------------------------------------------------- */
/* GREEN — every shape a healthy core really comes in                         */
/* ------------------------------------------------------------------------- */

static void healthy_cases(void)
{
    ft_tt_core_processor_liveness slots[FT_TT_PROC_COUNT];
    ft_tt_core_readiness readiness;
    ft_tt_core_status status;

    printf("\nHEALTHY — a guard that refuses any of these takes the model to the CPU forward\n");

    /* 1. A cold card. No queue was ever placed here, so the slots mean nothing
     *    and are not even read. */
    status_held(&status);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(readiness.ready, "cold card, all five held, no queue — READY");

    /* 2. Firmware up and idle with no queue: what a finished ft_tt_core probe
     *    leaves behind, and what a bind_core that never started leaves. */
    status_running(&status, FT_TT_RUN_MSG_DONE);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(readiness.ready, "firmware idle, no queue control block — READY");

    /* 3. THE RE-ARM. A residency that retired cleanly: firmware idle, the queue
     *    block still carries our magic, every slot says EXIT and no heartbeat
     *    moves. This is the ordinary second arm and it must not be refused. */
    status_running(&status, FT_TT_RUN_MSG_DONE);
    with_queue(&status, false);
    slots_quiet(slots, (uint32_t)FT_TT_QUEUE_STATE_EXIT);
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(readiness.ready, "retired residency: idle, all slots EXIT, nothing ticking — READY");
    check(readiness.ticking_mask == 0u, "  and nothing is reported ticking");

    /* 4. THE STALE-WORD TRAP. The memory family enables DM0|DM1 only, so its
     *    three TRISCs never run the program and never write EXIT — their slots
     *    hold whatever the last program left. Refusing on a state word rather
     *    than on a moving heartbeat would refuse this healthy core. */
    status_running(&status, FT_TT_RUN_MSG_DONE);
    with_queue(&status, false);
    slots_quiet(slots, (uint32_t)FT_TT_QUEUE_STATE_EXIT);
    slots[FT_TT_PROC_MATH0].state = (uint32_t)FT_TT_QUEUE_STATE_CB_WAIT_FRONT;
    slots[FT_TT_PROC_MATH1].state = (uint32_t)FT_TT_QUEUE_STATE_EXEC;
    slots[FT_TT_PROC_MATH2].state = 0xDEADBEEFu;
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(readiness.ready,
          "memory family's stale TRISC state words, no heartbeat moving — READY");

    /* 5. The debug register did not answer. UNREAD is not evidence of a wedge
     *    and must not be read as one. */
    status_running(&status, FT_TT_RUN_MSG_DONE);
    status.state.instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_UNREAD;
    status.verdict = ft_tt_core_status_classify(&status);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(readiness.ready, "instrn_buf UNREAD (register did not answer) — READY, not a wedge");
}

/* ------------------------------------------------------------------------- */
/* RED — the leftover, and everything else that must be refused               */
/* ------------------------------------------------------------------------- */

static void refusal_cases(void)
{
    ft_tt_core_processor_liveness slots[FT_TT_PROC_COUNT];
    ft_tt_core_readiness readiness;
    ft_tt_core_status status;

    printf("\nREFUSED — booting onto any of these is the wedge\n");

    /* THE BUG, EXACTLY AS SILICON PRODUCED IT. A refused descriptor: the reader
     * on DM0 aborted and wrote EXIT, so the CONTROL heartbeat is frozen and the
     * status read alone calls the core quiet — while TRISC0 spins out its bound
     * and its OWN heartbeat moves. Reader exited, compute thread has not. */
    status_running(&status, FT_TT_RUN_MSG_GO);
    with_queue(&status, false);
    slots_quiet(slots, (uint32_t)FT_TT_QUEUE_STATE_EXIT);
    slots[FT_TT_PROC_MATH0].state = (uint32_t)FT_TT_QUEUE_STATE_CB_WAIT_FRONT;
    slots[FT_TT_PROC_MATH0].ticking = true;
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready, "THE LEFTOVER: reader exited, TRISC0 heartbeat moving — REFUSED");
    check(readiness.state == FT_TT_CORE_NOT_READY_STILL_EXECUTING,
          "  and it is named STILL_EXECUTING, not a generic failure");
    check(readiness.waiting_can_help, "  and waiting can help — the spin is bounded");
    check(readiness.ticking_mask == (1u << FT_TT_PROC_MATH0),
          "  and the mask names TRISC0 alone");
    check(!status.queue_heartbeat_ticking,
          "  CONTROL heartbeat is frozen — the status read alone would have passed this core");

    /* The same leftover after the go signal has gone DONE. A heartbeat outranks
     * every verdict, including the two bootable ones, and this is the case that
     * makes that ordering load-bearing rather than decorative. */
    status_running(&status, FT_TT_RUN_MSG_DONE);
    with_queue(&status, false);
    slots_quiet(slots, (uint32_t)FT_TT_QUEUE_STATE_EXIT);
    slots[FT_TT_PROC_MATH0].ticking = true;
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready && readiness.state == FT_TT_CORE_NOT_READY_STILL_EXECUTING,
          "FIRMWARE_IDLE with a moving heartbeat — REFUSED, the heartbeat outranks the verdict");

    /* THE 0x66 WEDGE. Already wedged, and no amount of waiting fixes it: rung 1
     * does not clear it and only a card reset does. */
    status_running(&status, FT_TT_RUN_MSG_DONE);
    status.state.instrn_buf_status = 0x66u;
    status.verdict = ft_tt_core_status_classify(&status);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready && readiness.state == FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO,
          "instrn_buf 0x66 — REFUSED as the instruction-FIFO wedge");
    check(!readiness.waiting_can_help, "  and waiting CANNOT help — it needs a card reset");

    /* 0x66 while something is still ticking. The FIFO reading wins, because it
     * is the one that says the wait would be dead time before an inevitable
     * refusal. */
    status_running(&status, FT_TT_RUN_MSG_GO);
    status.state.instrn_buf_status = 0x66u;
    status.verdict = ft_tt_core_status_classify(&status);
    with_queue(&status, false);
    slots_quiet(slots, (uint32_t)FT_TT_QUEUE_STATE_EXEC);
    slots[FT_TT_PROC_MATH0].ticking = true;
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(readiness.state == FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO && !readiness.waiting_can_help,
          "0x66 outranks a moving heartbeat — no wait before an inevitable refusal");

    /* A launched kernel with nothing making progress. Not something to wait for
     * and not something to boot onto. */
    status_running(&status, FT_TT_RUN_MSG_GO);
    with_queue(&status, false);
    slots_quiet(slots, (uint32_t)FT_TT_QUEUE_STATE_EXEC);
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready && readiness.state == FT_TT_CORE_NOT_READY_KERNEL_STALLED,
          "go GO, nothing ticking — REFUSED as a stalled kernel");
    check(!readiness.waiting_can_help, "  and waiting cannot help");

    /* A kernel running with no queue block at all: a non-queue program. Same
     * refusal, reached without any slot being read. */
    status_running(&status, FT_TT_RUN_MSG_GO);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready, "go GO with no queue control block — REFUSED");

    status_running(&status, FT_TT_RUN_MSG_INIT);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready && readiness.state == FT_TT_CORE_NOT_READY_BOOT_STUCK,
          "BOOT_INCOMPLETE — REFUSED as a stuck boot");
    check(!readiness.waiting_can_help, "  and waiting cannot help");

    memset(&status, 0, sizeof(status));
    status.read_result = FT_TT_EIO;
    status.verdict = ft_tt_core_status_classify(&status);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(!readiness.ready && readiness.state == FT_TT_CORE_NOT_READY_NO_ANSWER,
          "a core that does not answer — REFUSED");
    check(!readiness.waiting_can_help, "  and waiting cannot help");

    /* MIXED: some running, some held, and nothing above matches. Report, do not
     * guess, and certainly do not boot. */
    memset(&status, 0, sizeof(status));
    status.read_result = FT_TT_OK;
    status.state.instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_READY;
    status.state.reset_register = ft_tt_processor_reset_bit(FT_TT_PROC_MATH2);
    status.state.go_messages[0].signal = FT_TT_RUN_MSG_DONE;
    status.verdict = ft_tt_core_status_classify(&status);
    memset(slots, 0, sizeof(slots));
    ft_tt_core_readiness_classify(&status, slots, &readiness);
    check(status.verdict == FT_TT_CORE_VERDICT_MIXED, "one TRISC held, the rest running is MIXED");
    check(!readiness.ready && readiness.state == FT_TT_CORE_NOT_READY_MIXED,
          "MIXED — REFUSED");
}

/* ------------------------------------------------------------------------- */
/* Every state has a string, and none of them is the fallthrough              */
/* ------------------------------------------------------------------------- */

static void string_cases(void)
{
    static const ft_tt_core_readiness_state all[] = {
        FT_TT_CORE_READY,
        FT_TT_CORE_NOT_READY_NO_ANSWER,
        FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO,
        FT_TT_CORE_NOT_READY_STILL_EXECUTING,
        FT_TT_CORE_NOT_READY_KERNEL_STALLED,
        FT_TT_CORE_NOT_READY_BOOT_STUCK,
        FT_TT_CORE_NOT_READY_MIXED
    };
    unsigned index;
    bool all_named = true;

    printf("\nEvery readiness state names itself\n");
    for (index = 0; index < sizeof(all) / sizeof(all[0]); index++) {
        const char *text = ft_tt_core_readiness_string(all[index]);

        if (!text || !*text || strcmp(text, "?") == 0)
            all_named = false;
    }
    check(all_named, "no readiness state falls through to \"?\"");
}

int main(void)
{
    printf("ft_tt_core_readiness — off-hardware classifier suite\n");
    printf("====================================================\n\n");

    verdict_cases();
    healthy_cases();
    refusal_cases();
    string_cases();

    printf("\n%u checks, %u failed\n", checks_run, checks_failed);
    return checks_failed == 0u ? 0 : 1;
}
