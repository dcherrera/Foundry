/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_readiness.c
 * @brief Implementation of the may-I-boot-here question (see ft_tt_core_readiness.h).
 *
 * @status New 2026-08-24. Classifier covered off-hardware by tt/tests/test_core_readiness.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_core_readiness.h"

#include <string.h>

#include "ft_tt_launch.h"
#include "ft_tt_memmap.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_queue.h"

const char *ft_tt_core_readiness_string(ft_tt_core_readiness_state state)
{
    switch (state) {
    case FT_TT_CORE_READY:
        return "ready";
    case FT_TT_CORE_NOT_READY_NO_ANSWER:
        return "the core does not answer a mailbox read";
    case FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO:
        return "the Tensix instruction FIFO is not ready — the wedge, and only a card reset "
               "clears it";
    case FT_TT_CORE_NOT_READY_STILL_EXECUTING:
        return "a processor is still executing — a heartbeat moved between two reads";
    case FT_TT_CORE_NOT_READY_KERNEL_STALLED:
        return "a kernel is launched and no processor is making progress";
    case FT_TT_CORE_NOT_READY_BOOT_STUCK:
        return "the core is stuck in its boot handshake";
    case FT_TT_CORE_NOT_READY_MIXED:
        return "the RISC-Vs are in a mixed reset state nothing here can name";
    }
    return "?";
}

/** Did any processor's heartbeat move? Fills `out->ticking_mask` on the way past. */
static bool anything_ticking(const ft_tt_core_processor_liveness *processor,
                             ft_tt_core_readiness *out)
{
    unsigned index;

    out->ticking_mask = 0u;
    if (!processor)
        return false;
    for (index = 0; index < FT_TT_PROC_COUNT; index++)
        if (processor[index].slot_read && processor[index].ticking)
            out->ticking_mask |= 1u << index;
    return out->ticking_mask != 0u;
}

void ft_tt_core_readiness_classify(const ft_tt_core_status *status,
                                   const ft_tt_core_processor_liveness *processor,
                                   ft_tt_core_readiness *out)
{
    bool ticking;

    if (!status || !out)
        return;
    memset(out, 0, sizeof(*out));
    out->status = *status;
    if (processor)
        memcpy(out->processor, processor, sizeof(out->processor));

    /* THE MAILBOX FIRST. Everything below reads words the mailbox snapshot brought back, so
     * a failed read makes all of them meaningless rather than false. */
    if (status->read_result != FT_TT_OK) {
        out->state = FT_TT_CORE_NOT_READY_NO_ANSWER;
        return;
    }

    /* THE INSTRUCTION FIFO BEFORE THE VERDICT, because it is the one reading that says
     * WAITING CANNOT HELP, and it is true whatever the verdict says. BRISC firmware's
     * device_setup() pushes into this FIFO before it releases the subordinates, so a boot
     * onto a core whose FIFO cannot accept a write stalls BRISC inside a single store —
     * which is the wedge itself, not a symptom of it. MEASURED 2026-08-17 over 141 cores:
     * healthy is 0x77 in every state seen; the wedged core reads 0x66 and keeps reading it
     * across further park/assert cycles. UNREAD (0xFFFFFFFF) means the debug register did
     * not answer, which is not evidence of a wedge and is not treated as one. */
    if (status->state.instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_READY &&
        status->state.instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_UNREAD) {
        out->state = FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO;
        (void)anything_ticking(processor, out);
        return;
    }

    ticking = anything_ticking(processor, out);

    /* A TICKING HEARTBEAT OUTRANKS EVERY VERDICT, INCLUDING THE TWO BOOTABLE ONES.
     *
     * This is the whole reason the module exists. The verdict rests on the go signal and the
     * reset register, and the failure being guarded against is a compute thread spinning
     * while the reader that owns those words has already left. A moving heartbeat is a
     * first-hand statement that a processor is executing; it is checked before anything that
     * would let the core through. It also self-clears — the spin is bounded — so this is the
     * one not-ready state a caller may wait out. */
    if (ticking) {
        out->state = FT_TT_CORE_NOT_READY_STILL_EXECUTING;
        out->waiting_can_help = true;
        return;
    }

    switch (status->verdict) {
    case FT_TT_CORE_VERDICT_HELD:
    case FT_TT_CORE_VERDICT_FIRMWARE_IDLE:
        /* THE TWO STATES FIRMWARE MAY BE BOOTED ONTO, and they are exactly the two states a
         * clean shutdown leaves behind: HELD is cold or parked, FIRMWARE_IDLE is what a
         * retired residency and a finished probe both leave (ft_tt_core_probe.h: "firmware
         * running and idle, NOT parked"). Nothing else is on this list. */
        out->state = FT_TT_CORE_READY;
        out->ready = true;
        return;
    case FT_TT_CORE_VERDICT_KERNEL_RUNNING:
        /* GO in the mailbox with nothing ticking. Either a non-queue kernel is running, or a
         * queue kernel is somewhere that does not bump a heartbeat. Neither is something to
         * wait for, and neither is something to boot onto. */
        out->state = FT_TT_CORE_NOT_READY_KERNEL_STALLED;
        return;
    case FT_TT_CORE_VERDICT_BOOT_INCOMPLETE:
    case FT_TT_CORE_VERDICT_SYNC_BLOCKED:
        out->state = FT_TT_CORE_NOT_READY_BOOT_STUCK;
        return;
    case FT_TT_CORE_VERDICT_NO_ANSWER:
        out->state = FT_TT_CORE_NOT_READY_NO_ANSWER;
        return;
    case FT_TT_CORE_VERDICT_MIXED:
        break;
    }
    out->state = FT_TT_CORE_NOT_READY_MIXED;
}

/**
 * Sample the five per-processor queue status slots TWICE and report what moved.
 *
 * The slots are at a fixed offset from `FT_TT_OP_QUEUE_BASE`, which is where every residency
 * in this stack puts its control block — so a process that did not create the queue can
 * still read them. They are consulted only when the control block carries our magic;
 * otherwise the words belong to nobody and mean nothing.
 */
static void read_processor_liveness(const ft_tt_noc_access *noc, const ft_tt_core_status *status,
                                    ft_tt_core_processor_liveness *out)
{
    unsigned processor;

    memset(out, 0, FT_TT_PROC_COUNT * sizeof(*out));
    if (!status->queue_present)
        return;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        const uint32_t address = FT_TT_QUEUE_STATUS_ADDRESS(FT_TT_OP_QUEUE_BASE, processor);
        uint32_t first[FT_TT_QUEUE_STATUS_WORDS];
        uint32_t again[FT_TT_QUEUE_STATUS_WORDS];

        if (noc->read(noc->user, status->x, status->y, address, first, sizeof(first)) != FT_TT_OK)
            continue;
        if (noc->read(noc->user, status->x, status->y, address, again, sizeof(again)) != FT_TT_OK)
            continue;
        out[processor].slot_read = true;
        out[processor].state = first[FT_TT_QUEUE_STATUS_STATE];
        out[processor].heartbeat = first[FT_TT_QUEUE_STATUS_HEARTBEAT];
        out[processor].ticking =
            again[FT_TT_QUEUE_STATUS_HEARTBEAT] != first[FT_TT_QUEUE_STATUS_HEARTBEAT];
    }
}

int ft_tt_core_readiness_read(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                              uint32_t index, ft_tt_core_readiness *out)
{
    ft_tt_core_processor_liveness processor[FT_TT_PROC_COUNT];
    ft_tt_core_status status;
    int rc;

    if (!noc || !noc->read || !grid || !out)
        return FT_TT_EINVAL;

    rc = ft_tt_core_status_read(noc, grid, index, &status);
    if (rc != FT_TT_OK)
        return rc;

    memset(processor, 0, sizeof(processor));
    if (status.read_result == FT_TT_OK)
        read_processor_liveness(noc, &status, processor);
    ft_tt_core_readiness_classify(&status, processor, out);
    return FT_TT_OK;
}
