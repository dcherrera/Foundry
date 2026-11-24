/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_norm_role_reader.c
 * @brief Reader third of the PERSISTENT NORM ROLE (resident_norm_role_protocol.h): per step,
 *        the bridge precondition, this member's input slice, the step's constants (prefetched
 *        one step ahead from the schedule table), and the exchange — slice j of SqAll is
 *        pushed once member j's flag reaches this step's sequence.
 *
 * THIS PROCESSOR OWNS `tail`, THE WATCHDOG WORD AND THE PHASE CLOCKS (kNrWordInputRead,
 * kNrWordFlagsIn). Descriptor order is the queue's: a step whose index is not the one the role expects is refused (resident_norm_role_protocol.h
 * ftnr_descriptor_check), and so is a schedule table that does not check, before any input
 * is touched.
 *
 * READS, IN CRITICAL-PATH ORDER:
 *   1. the input slice, ONE contiguous read of S whole tiles (rows 1-31 of every activation
 *      strip are zero, so whole tiles are today's ADD operand bit for bit);
 *   2. the constants, only if the previous step did not prefetch them (the first step);
 *   3. the exchange: nothing is read, the other members WRITE SqAll; this thread only turns
 *      their flags into CB pages;
 *   4. after the step's pages are pushed, the NEXT step's constants (scaler, eps, gamma slice)
 *      into the second half of their CBs — off the critical path, ahead of the next
 *      precondition.
 *
 * Every wait is stoppable and bounded (rule 3); on an abort the CB pushes still balance, so
 * the compute and writer threads finish the step and the role retires.
 *
 * Compile-time args: 0 tile bytes (2048), 1 wait mode (1 stoppable, ships).
 * Runtime args: the ten of resident_queue_protocol.h.
 *
 * @status P0 (P0.2). Card gate pending.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_norm_role_protocol.h"
#include "resident_precondition.h"
#include "resident_queue_protocol.h"

#define TILE_BYTES ft_tensix_compile_time_arg(0)
#define WAIT_MODE ft_tensix_compile_time_arg(1)

/** What the reader carries from one step to the next. */
typedef struct norm_role_reader {
    volatile FTQ_L1_PTR uint32_t *table;
    volatile FTQ_L1_PTR uint32_t *flags;
    uint32_t member;
    uint32_t sequence;        /* steps run since arm */
    uint32_t prefetched_step; /* the step whose constants already sit in their CBs */
    bool prefetched;
    bool checked;             /* the table passed ftnr_table_check */
} norm_role_reader;

/** Refuse the step: record why (word kNrWordReason) and abort the descriptor. */
static void refuse_step(ftq_context *ctx, uint32_t reason)
{
    ctx->control[kNrWordReason] = reason;
    ftq_abort(ctx, kErrBadParameters);
}

/** `tiles` consecutive DRAM tiles from tile `first` of (xy, base) into `cb`, one read. */
static bool read_tiles(ftq_context *ctx, uint32_t cb, uint32_t xy, uint32_t base, uint32_t first,
                       uint32_t tiles)
{
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)cb, (int32_t)tiles, WAIT_MODE))
        return false;
    if (!ctx->aborting) {
        ft_tensix_noc_read(ft_tensix_noc_address(ftnr_x(xy), ftnr_y(xy), base + first * TILE_BYTES),
                           ft_tensix_cb_write_pointer(cb), tiles * TILE_BYTES);
        ft_tensix_noc_read_barrier();
    }
    ft_tensix_cb_push_back((int32_t)cb, (int32_t)tiles);
    return true;
}

/** Step `step`'s constants: the scaler and eps tiles, then this member's gamma slice. */
static bool read_constants(ftq_context *ctx, const norm_role_reader *role, uint32_t step)
{
    const volatile FTQ_L1_PTR uint32_t *entry = ftnr_entry(role->table, step);
    const uint32_t xy = entry[kNrEntryConstXy];
    const uint32_t base = entry[kNrEntryConstBase];
    const uint32_t slice = role->table[kNrHeadSliceTiles];

    return read_tiles(ctx, kNrCbScaler, xy, base, 0u, 1u) &&
           read_tiles(ctx, kNrCbEps, xy, base, 1u, 1u) &&
           read_tiles(ctx, kNrCbGamma, xy, base, 2u + role->member * slice, slice);
}

/**
 * At boot: SqAll and the flags page (flags, stage words, arrival clocks) to zero. Faces 2-3
 * of every SqAll page must read as today's cb_mid does (zero; whole-page writes carry that
 * zero, faces-0-1 writes leave it); a flag left by an earlier process would satisfy a wait
 * before its slice landed.
 */
static void zero_exchange(void)
{
    volatile FTQ_L1_PTR uint32_t *sqall =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbSqAll);
    volatile FTQ_L1_PTR uint32_t *flags =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbFlags);
    const uint32_t sqall_words = ft_tensix_cb(kNrCbSqAll)->fifo_size / 4u;

    for (uint32_t w = 0; w < sqall_words; w++)
        sqall[w] = 0u;
    for (uint32_t w = 0; w < (uint32_t)kNrFlagsBytes / 4u; w++)
        flags[w] = 0u;
}

/** Bump the watchdog word (this thread's alone). */
static void watchdog(ftq_context *ctx)
{
    ctx->control[kNrWordWatchdog] = ctx->control[kNrWordWatchdog] + 1u;
}

/** Add the time since this descriptor's precondition stamp to role word `word`. */
static void phase_clock(ftq_context *ctx, uint32_t word)
{
    const volatile FTQ_L1_PTR uint32_t *stamps =
        (const volatile FTQ_L1_PTR uint32_t *)(uintptr_t)((uint32_t)(uintptr_t)ctx->control +
                                                          (uint32_t)kPreStampOffset);

    ctx->control[word] +=
        (ft_bh_wall_clock_low() - stamps[ctx->cursor & ((uint32_t)kPreStampSlots - 1u)]) >> 4;
}

/**
 * Arrival clocks (spec §11.13): the first poll that sees member k's flag at this step adds the
 * time since this member's precondition stamp to flags word kNrFlagArrival + k. Every unseen
 * flag is looked at on every poll, so a slice that lands before an earlier one is timed when it
 * lands. Against each sender's own flag-sent clock this splits precondition skew from send time.
 */
static uint32_t note_arrivals(ftq_context *ctx, const norm_role_reader *role, uint32_t members,
                              uint32_t seen)
{
    const volatile FTQ_L1_PTR uint32_t *stamps =
        (const volatile FTQ_L1_PTR uint32_t *)(uintptr_t)((uint32_t)(uintptr_t)ctx->control +
                                                          (uint32_t)kPreStampOffset);

    const uint32_t all = (1u << members) - 1u;
    const uint32_t before = seen;

    for (uint32_t k = 0; k < members; k++)
        if ((seen & (1u << k)) == 0u && ftnr_flag_ready(role->flags[k], role->sequence)) {
            seen |= 1u << k;
            role->flags[kNrFlagArrival + k] +=
                (ft_bh_wall_clock_low() -
                 stamps[ctx->cursor & ((uint32_t)kPreStampSlots - 1u)]) >> 4;
        }
    if (seen == all && before != all)
        role->flags[kNrFlagAllSeen] +=
            (ft_bh_wall_clock_low() - stamps[ctx->cursor & ((uint32_t)kPreStampSlots - 1u)]) >> 4;
    return seen;
}

/**
 * Turn every member's flag for this step into SqAll pages, slice by slice in member order,
 * so the compute thread's reduce runs over slice 0 while later slices are still landing.
 * Bounded and stoppable; an abort still pushes every slice (rule 3).
 */
static bool push_exchange(ftq_context *ctx, const norm_role_reader *role)
{
    const uint32_t members = role->table[kNrHeadMembers] & 0xFFu;
    const uint32_t slice = role->table[kNrHeadSliceTiles];
    uint32_t seen = 0u;
    bool ok = true;

    for (uint32_t j = 0; j < members; j++) {
        uint32_t spins = 0u;

        while (ok && !ctx->aborting) {
            uint32_t action;

            ft_bh_l1_cache_invalidate();
            seen = note_arrivals(ctx, role, members, seen);
            action = ftnr_exchange_wait_action(role->flags[j], role->sequence, spins,
                                               ftq_stop_requested(ctx), ctx->cb_spin_limit);
            if (action == (uint32_t)kNrWaitReady)
                break;
            if (action == (uint32_t)kNrWaitStop) {
                ftq_abort(ctx, kErrNone);
                ok = false;
            } else if (action == (uint32_t)kNrWaitTimeout) {
                ftq_abort(ctx, kErrCbWaitTimeout);
                ok = false;
            } else if ((++spins & 1023u) == 0u) {
                watchdog(ctx);
            }
        }
        const uint32_t reserve_from = ft_bh_wall_clock_low();

        (void)ftq_cb_reserve_back_stoppable(ctx, (int32_t)kNrCbSqAll, (int32_t)slice, WAIT_MODE);
        role->flags[kNrFlagSqAllWait] += (ft_bh_wall_clock_low() - reserve_from) >> 4;
        ft_tensix_cb_push_back((int32_t)kNrCbSqAll, (int32_t)slice);
    }
    return ok && !ctx->aborting;
}

/** One step's reads. False: refused or abandoned — the caller leaves the loop unconsumed. */
static bool read_one_step(ftq_context *ctx, norm_role_reader *role,
                          volatile FTQ_L1_PTR uint32_t *parameters)
{
    uint32_t verdict, step, slice;
    const volatile FTQ_L1_PTR uint32_t *entry;

    if (!role->checked) {
        /* The member index is the table's: the host writes one table per member. */
        role->member = (role->table[kNrHeadMembers] >> 8) & 0xFFu;
        verdict = ftnr_table_check(role->table, role->member);
        if (verdict != (uint32_t)kNrOk) {
            refuse_step(ctx, verdict);
            return false;
        }
        role->checked = true;
    }
    role->sequence++;
    verdict = ftnr_descriptor_check(parameters, role->table, role->sequence);
    if (verdict != (uint32_t)kNrOk) {
        refuse_step(ctx, verdict);
        return false;
    }
    step = parameters[kNrParamStep];
    entry = ftnr_entry(role->table, step);
    slice = role->table[kNrHeadSliceTiles];

    if (!read_tiles(ctx, kNrCbIn, entry[kNrEntryInputXy], entry[kNrEntryInputBase],
                    role->member * slice, slice))
        return false;
    if (!(role->prefetched && role->prefetched_step == step) &&
        !read_constants(ctx, role, step))
        return false;
    phase_clock(ctx, kNrWordInputRead);
    if (!push_exchange(ctx, role))
        return false;
    phase_clock(ctx, kNrWordFlagsIn);
    watchdog(ctx);

    /* The next step's constants, behind everything this step needed. */
    role->prefetched_step = ftnr_next_step(step, role->table[kNrHeadSteps]);
    role->prefetched = read_constants(ctx, role, role->prefetched_step);
    return role->prefetched && !ctx->aborting;
}

void kernel_main(void)
{
    ftq_context ctx;
    norm_role_reader role;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    ctx.control[kFieldState] = kStateBoot;
    if (!ftq_check_control(&ctx) || ctx.control[kFieldTail] != ctx.control[kFieldCompletedCount]) {
        if (ctx.error == kErrNone)
            ctx.error = kErrBadParameters;
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    role.table = (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbTable);
    role.flags = (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbFlags);
    role.member = 0u;
    role.sequence = 0u;
    role.prefetched_step = 0u;
    role.prefetched = false;
    role.checked = false;
    zero_exchange();
    ctx.control[kNrWordWatchdog] = 0u;
    ctx.control[kNrWordInputRead] = 0u;
    ctx.control[kNrWordFlagsIn] = 0u;
    ctx.control[kNrWordReason] = kNrOk;
    ctx.cursor = ctx.control[kFieldCompletedCount];

    while (running) {
        volatile FTQ_L1_PTR uint32_t *descriptor;
        uint32_t opcode;

        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;
        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        /* BRIDGE: today's S2 precondition, before any input is read. */
        if (!ftq_precondition_wait(&ctx)) {
            running = false;
            continue;
        }
        descriptor = ftq_descriptor(&ctx);
        opcode = descriptor[kDescOpcode];
        ctx.control[kFieldLastOpcode] = opcode;
        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpReduce) {
            if (!read_one_step(&ctx, &role, ftq_parameters(descriptor))) {
                running = false;
                continue;
            }
        } else if (opcode == kOpStop) {
            running = false;
        } else if (opcode != kOpNop) {
            ctx.error = kErrBadOpcode;
            running = false;
            continue;
        }
        ctx.cursor++;
        ctx.control[kFieldTail] = ctx.cursor;
    }
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_write_barrier();
    ctx.control[kFieldError] = ctx.error;
    ctx.control[kFieldState] = kStateExit;
    ftq_status_set(&ctx, kStateExit);
}
