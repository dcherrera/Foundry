/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_status_print.c
 * @brief Implementation of the three human-readable forms of an ft_tt_core_status.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#include "ft_tt_core_status_print.h"

#include <stdio.h>

#include "ft_tt_launch.h"

/** Bit for each processor in the soft-reset register, in FT_TT_PROC_* order. */
static const char *const PROCESSOR_SHORT_NAME[FT_TT_PROC_COUNT] = {"BR", "NC", "T0", "T1", "T2"};

/** The library's predicate under this file's old local name. */
#define processor_is_held ft_tt_core_status_processor_is_held

void ft_tt_core_status_print_header(void)
{
    printf("%5s %9s  %-10s %-29s %-10s %s\n", "core", "noc(x,y)", "reset", "BR NC T0 T1 T2",
           "instrnbuf", "verdict");
}

void ft_tt_core_status_print_line(const ft_tt_core_status *status)
{
    char processors[32];
    unsigned processor;
    size_t used = 0;

    if (!status)
        return;
    if (status->read_result != FT_TT_OK) {
        printf("%5u   (%2u,%2u)  %-10s %-29s %-10s %s (rc=%d)\n", status->index, status->x,
               status->y, "-", "-", "-", ft_tt_core_verdict_string(status->verdict),
               status->read_result);
        return;
    }

    processors[0] = '\0';
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        int written = snprintf(processors + used, sizeof(processors) - used, "%s%s",
                               processor ? " " : "",
                               processor_is_held(status->state.reset_register, processor) ? "hold"
                                                                                         : "run ");
        if (written < 0 || (size_t)written >= sizeof(processors) - used)
            break;
        used += (size_t)written;
    }

    printf("%5u   (%2u,%2u)  0x%08x %-29s 0x%08x %s%s%s\n", status->index, status->x, status->y,
           status->state.reset_register, processors, status->state.instrn_buf_status,
           ft_tt_core_verdict_string(status->verdict),
           status->state.instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_READY ||
                   status->state.instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_UNREAD
               ? ""
               : " [Tensix FIFO NOT READY]",
           status->queue_present
               ? (status->queue_heartbeat_ticking ? " [queue ALIVE]" : " [queue not ticking]")
               : "");
}

void ft_tt_core_status_print_detail(const ft_tt_core_status *status)
{
    unsigned processor;

    if (!status)
        return;

    printf("Tensix %u — translated NoC (%u, %u)\n", status->index, status->x, status->y);
    if (status->read_result != FT_TT_OK) {
        printf("  the core does not answer a mailbox read (rc=%d). This is past what a host "
               "write can fix.\n", status->read_result);
        return;
    }

    printf("  soft reset 0x%08x:", status->state.reset_register);
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++)
        printf(" %s=%s", PROCESSOR_SHORT_NAME[processor],
               processor_is_held(status->state.reset_register, processor) ? "held" : "running");
    printf("\n");

    printf("  subordinate sync  NCRISC=0x%02x TRISC0=0x%02x TRISC1=0x%02x TRISC2=0x%02x"
           "   (0x00 done, 0x40 init, 0x80 go)\n",
           status->state.subordinate_sync[0], status->state.subordinate_sync[1],
           status->state.subordinate_sync[2], status->state.subordinate_sync[3]);
    printf("  go[0].signal=0x%02x  go_message_index=%u  launch_msg_rd_ptr=%u\n",
           status->state.go_messages[0].signal, status->state.go_message_index,
           status->state.launch_msg_rd_ptr);
    printf("  instrn_buf_status=0x%08x  invalid_instrn=0x%08x   (%s)\n",
           status->state.instrn_buf_status, status->state.invalid_instrn,
           status->state.instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_READY
               ? "Tensix FIFO ready"
               : (status->state.instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_UNREAD
                      ? "unread"
                      : "Tensix FIFO NOT READY"));
    printf("  launch: enables=0x%08x mode=%u config_base=0x%08x\n",
           status->state.kernel_config.enables, status->state.kernel_config.mode,
           status->state.kernel_config.kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX]);

    if (status->queue_present) {
        printf("  resident queue: state=%s head=%u tail=%u completed=%u error=%s stop=%u "
               "heartbeat=%u (%s)\n",
               ft_tt_queue_state_string(status->queue_control[FT_TT_QUEUE_FIELD_STATE]),
               status->queue_control[FT_TT_QUEUE_FIELD_HEAD],
               status->queue_control[FT_TT_QUEUE_FIELD_TAIL],
               status->queue_control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
               ft_tt_queue_kernel_error_string(status->queue_control[FT_TT_QUEUE_FIELD_ERROR]),
               status->queue_control[FT_TT_QUEUE_FIELD_STOP_REQUEST],
               status->queue_control[FT_TT_QUEUE_FIELD_HEARTBEAT],
               status->queue_heartbeat_ticking ? "ALIVE" : "not ticking");
    } else {
        printf("  resident queue: none — no resident kernel left a control block at 0x%08x\n",
               (unsigned)FT_TT_OP_QUEUE_BASE);
    }

    printf("  verdict: %s\n", ft_tt_core_verdict_string(status->verdict));
    if (status->verdict == FT_TT_CORE_VERDICT_BOOT_INCOMPLETE)
        printf("           BRISC is out of reset and never wrote RUN_MSG_DONE — this core "
               "did not finish booting\n");
    /* MEASURED 2026-08-17, 141 cores: healthy reads 0x77 with invalid_instrn 0 in every
     * state seen (running idle, running post-broadcast, held in reset, and straight after
     * an ASIC reset); the core the 1x5 multicast relaunch arm wedged reads 0x66 with
     * invalid_instrn 1, and keeps reading it across further park/assert cycles. This is
     * the first reading that separates a wedged core from a healthy one WITHOUT booting
     * it — see the note in ft_tt_launch.h. It is a detector, NOT a predictor: the same
     * core read 0x77 both immediately before and immediately after the park that preceded
     * the failed boot. */
    if (status->state.instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_READY &&
        status->state.instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_UNREAD)
        printf("           the Tensix instruction FIFO is not ready (0x%08x, ready is "
               "0x%02x). BRISC firmware's device_setup() pushes into it before it releases "
               "the subordinates, so this core cannot complete a firmware boot. Only a card "
               "reset clears it: `recover --rung 3 --driver udrv --write --then-probe`\n",
               status->state.instrn_buf_status, (unsigned)FT_TT_TENSIX_INSTRN_BUF_READY);
    else if (status->verdict == FT_TT_CORE_VERDICT_HELD)
        printf("           a read cannot tell a parked healthy core from a parked wedged one "
               "on the mailbox alone; instrn_buf_status is ready, which is evidence but not "
               "proof — run `probe` to find out\n");
}
