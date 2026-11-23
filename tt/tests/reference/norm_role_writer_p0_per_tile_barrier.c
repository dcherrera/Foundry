/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file norm_role_writer_p0_per_tile_barrier.c
 * @brief RED FIXTURE, never built: the P0 norm-role writer (foundry 56efb70d) frozen verbatim,
 *        with its per-tile flushed barrier on the exchange and the output. test_norm_role.c
 *        check 9 reads it under RED=per_tile_barrier and must fail.
 * @status Fixture.
 * @issues None
 * @todo None
 */
/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_norm_role_writer.c
 * @brief Writer third of the PERSISTENT NORM ROLE (resident_norm_role_protocol.h): per step,
 *        this member's x^2 slice to every member's SqAll and its flag, then the normed slice
 *        to the output strip (and its replicas), then TODAY'S completion — the vector
 *        writer's publish, word for word (S1 group pull on member 0, word 56, token, count).
 *
 * THE EXCHANGE. Page i of CB Sq goes, faces 0-1 only (kNrExchangeBytes), to SqAll page
 * member*S + i on every member: one multicast per rectangle target (the sender excluded by
 * the non-loopback multicast) and one unicast to this core. After every page, ONE write
 * barrier (every byte acked at every receiver), then the flag: this member's word of every
 * receiver's flags page set to the step's sequence by a semaphore multicast from stage word
 * ftnr_flag_stage_word(member) (same offset within 16 bytes as the destination word, or the
 * write does not land), and this core's own word by a plain store. A receiver therefore never
 * sees the flag before the data.
 *
 * THE OUTPUT. As resident_vector_writer.c (the replica map resolved once per step, one flushed
 * barrier per tile to pop, one ack barrier before publishing) but FACES 0-1 of each tile only
 * (kNrExchangeBytes), to every replica: the decode consumers of `normed` read row-0 half tiles
 * (kMatmulRow0), and faces 2-3 of the strip are the zero rows the shipped norm also leaves
 * zero. Half the bytes of the step's largest transfer (S tiles x 8 replicas).
 *
 * ROLE WORDS: kNrWordSteps (published steps), kNrWordBusy (cycles >> 4 from the reader's
 * precondition stamp to the publish) and kNrWordFlagSent (the same clock to this member's flag
 * sent) are this thread's alone.
 *
 * Compile-time args: 0 tile bytes (2048), 1 wait mode (1 stoppable, ships).
 * Runtime args: the ten of resident_queue_protocol.h.
 *
 * @status P0. Card gate pending.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "ft_tensix_noc_multicast.h"
#include "resident_group_pull.h"
#include "resident_norm_role_protocol.h"
#include "resident_output_replicas.h"
#include "resident_precondition_protocol.h"
#include "resident_queue_protocol.h"

#define TILE_BYTES ft_tensix_compile_time_arg(0)
#define WAIT_MODE ft_tensix_compile_time_arg(1)

/** Add the time from descriptor `index`'s precondition stamp (the reader's) to now to `word`. */
static void phase_clock(ftq_context *ctx, uint32_t word, uint32_t index)
{
    const volatile FTQ_L1_PTR uint32_t *stamps =
        (const volatile FTQ_L1_PTR uint32_t *)(uintptr_t)((uint32_t)(uintptr_t)ctx->control +
                                                          (uint32_t)kPreStampOffset);

    ctx->control[word] +=
        (ft_bh_wall_clock_low() - stamps[index & ((uint32_t)kPreStampSlots - 1u)]) >> 4;
}

/** Add this step's busy time: from the reader's precondition stamp to now. */
static void account_busy(ftq_context *ctx, uint32_t index)
{
    phase_clock(ctx, kNrWordBusy, index);
    ctx->control[kNrWordSteps] += 1u;
}

/** The multicast address of `address` over target `t`'s rectangle. */
static bool target_address(const volatile FTQ_L1_PTR uint32_t *table, uint32_t t,
                           uint32_t address, uint64_t *out)
{
    const uint32_t rect = table[kNrHeadTarget0 + 2u * t];

    return ft_tensix_multicast_address(rect & 0xFFu, (rect >> 8) & 0xFFu, (rect >> 16) & 0xFFu,
                                       (rect >> 24) & 0xFFu, address, out) ==
           FT_TENSIX_MULTICAST_OK;
}

/** Receivers of target `t`, the sender excluded. */
static uint32_t target_receivers(const volatile FTQ_L1_PTR uint32_t *table, uint32_t t)
{
    return table[kNrHeadTarget0 + 2u * t + 1u] & 0xFFu;
}

/** This member's x^2 slice into every member's SqAll, then its flag. */
static bool send_exchange(ftq_context *ctx, const volatile FTQ_L1_PTR uint32_t *table,
                          volatile FTQ_L1_PTR uint32_t *flags, uint32_t member,
                          uint32_t sequence)
{
    const uint32_t slice = table[kNrHeadSliceTiles];
    const uint32_t targets = table[kNrHeadTargets];
    const uint32_t self_xy = table[kNrHeadSelfXy];
    const uint32_t sqall = ft_tensix_cb_base_pointer(kNrCbSqAll);
    const uint32_t flags_base = (uint32_t)(uintptr_t)flags;
    uint64_t address;

    for (uint32_t i = 0; i < slice; i++) {
        const bool have = ftq_cb_wait_front_stoppable(ctx, (int32_t)kNrCbSq, 1, WAIT_MODE);

        if (have && !ctx->aborting) {
            const uint32_t source = ft_tensix_cb_read_pointer(kNrCbSq);
            const uint32_t landing =
                sqall + ftnr_sqall_offset(member, i, slice, TILE_BYTES);

            for (uint32_t t = 0; t < targets; t++)
                if (target_receivers(table, t) != 0u && target_address(table, t, landing, &address))
                    (void)ft_tensix_noc_write_multicast(source, address, kNrExchangeBytes,
                                                        target_receivers(table, t));
            ft_tensix_noc_write(source,
                                ft_tensix_noc_address(ftnr_x(self_xy), ftnr_y(self_xy), landing),
                                kNrExchangeBytes);
            ft_tensix_noc_writes_flushed_barrier();
        }
        ft_tensix_cb_pop_front((int32_t)kNrCbSq, 1);
    }
    if (ctx->aborting)
        return false;
    ft_tensix_noc_write_barrier();
    flags[ftnr_flag_stage_word(member)] = sequence;
    for (uint32_t t = 0; t < targets; t++)
        if (target_receivers(table, t) != 0u &&
            target_address(table, t, flags_base + 4u * member, &address))
            (void)ft_tensix_noc_semaphore_set_multicast(
                flags_base + 4u * ftnr_flag_stage_word(member), address,
                target_receivers(table, t));
    flags[member] = sequence;
    ft_tensix_noc_write_barrier();
    return true;
}

/** This member's normed slice to the output strip and its replicas. */
static bool write_output(ftq_context *ctx, const volatile FTQ_L1_PTR uint32_t *entry,
                         uint32_t member, uint32_t slice)
{
    const uint32_t xy = entry[kNrEntryOutXy];
    const uint32_t base = entry[kNrEntryOutBase];
    uint32_t destination_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t destination_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t destination_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    const uint32_t destinations =
        ftr_resolve_destinations((const volatile uint32_t *)(uintptr_t)FT_TT_REPLICA_TABLE_ADDRESS,
                                 ftnr_x(xy), ftnr_y(xy), base, destination_x, destination_y,
                                 destination_base);

    for (uint32_t i = 0; i < slice; i++) {
        const bool have = ftq_cb_wait_front_stoppable(ctx, (int32_t)kNrCbOut, 1, WAIT_MODE);

        if (have && !ctx->aborting) {
            const uint32_t source = ft_tensix_cb_read_pointer(kNrCbOut);
            const uint32_t offset = (member * slice + i) * TILE_BYTES;

            for (uint32_t copy = 0u; copy < destinations; copy++)
                ft_tensix_noc_write(source,
                                    ft_tensix_noc_address(destination_x[copy],
                                                          destination_y[copy],
                                                          destination_base[copy] + offset),
                                    kNrExchangeBytes);
            ft_tensix_noc_writes_flushed_barrier();
        }
        ft_tensix_cb_pop_front((int32_t)kNrCbOut, 1);
    }
    ft_tensix_noc_write_barrier();
    return !ctx->aborting;
}

/** One step's writes. The reader has refused anything malformed; the step index is still
 *  bounded here because this thread fetches on `head`, ahead of the reader's check. */
static bool write_one_step(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                           uint32_t sequence)
{
    const volatile FTQ_L1_PTR uint32_t *table =
        (const volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbTable);
    volatile FTQ_L1_PTR uint32_t *flags =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbFlags);
    const uint32_t member = (table[kNrHeadMembers] >> 8) & 0xFFu;
    const uint32_t step = parameters[kNrParamStep];

    if (table[kNrHeadMagic] != (uint32_t)kNrTableMagic || step >= table[kNrHeadSteps] ||
        member >= kNrMaxMembers || table[kNrHeadTargets] > kNrMaxTargets) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!send_exchange(ctx, table, flags, member, sequence))
        return false;
    phase_clock(ctx, kNrWordFlagSent, ctx->cursor);
    return write_output(ctx, ftnr_entry(table, step), member, table[kNrHeadSliceTiles]);
}


void kernel_main(void)
{
    ftq_context ctx;
    uint32_t completed;
    uint32_t sequence = 0u;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    ctx.control[kNrWordSteps] = 0u;
    ctx.control[kNrWordBusy] = 0u;
    ctx.control[kNrWordFlagSent] = 0u;
    completed = ctx.control[kFieldCompletedCount];
    ctx.cursor = completed;

    while (running) {
        volatile FTQ_L1_PTR uint32_t *descriptor;
        uint32_t opcode, token, group_sequence, pull;
        bool done = true;

        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;
        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        descriptor = ftq_descriptor(&ctx);
        opcode = descriptor[kDescOpcode];
        token = descriptor[kDescToken];
        group_sequence = descriptor[kDescA5];
        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpReduce) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_step(&ctx, ftq_parameters(descriptor), ++sequence);
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }
        if (!done) {
            running = false;
            continue;
        }
        /* Publishing waits on `tail`: the reader finished this descriptor. */
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }
        ctx.cursor++;
        pull = descriptor[kDescA3];
        if (opcode == kOpReduce && (pull & ~kGroupPullFlags) > 1u && (pull & kPullRedSkip) == 0u &&
            !ftq_group_pull(&ctx, pull & ~kGroupPullFlags, descriptor[kDescA4])) {
            running = false;
            continue;
        }
        ctx.control[kFieldGroupDone] = group_sequence;
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        if (opcode == kOpReduce)
            account_busy(&ctx, ctx.cursor - 1u);
        ftq_group_post(&ctx, group_sequence, token);
    }
    ftq_status_set(&ctx, kStateExit);
}
