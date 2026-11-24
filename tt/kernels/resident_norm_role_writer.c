/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_norm_role_writer.c
 * @brief Writer third of the PERSISTENT NORM ROLE (resident_norm_role_protocol.h): per step,
 *        this member's x^2 slice to every member's SqAll and its flag, then the normed slice
 *        to the output strip (and its replicas), then TODAY'S completion — the vector
 *        writer's publish, word for word (S1 group pull on member 0, word 56, token, count).
 *
 * THE EXCHANGE (spec §11.11 levers 1-2, §11.13). Once all S pages of CB Sq are in, the slice
 * goes to SqAll pages member*S .. member*S + S-1 on every member: per rectangle target (the
 * sender excluded by the non-loopback multicast) one whole-page multicast, or under
 * kNrExchangeFaces one faces-0-1 multicast a page (half the link bytes); targets in table order,
 * or under kNrExchangeStagger starting at target member mod targets; then one whole-run unicast
 * to this core. Per contiguous run of the Sq ring (ftnr_exchange_runs: one run, two only across a
 * wrap). Then ONE write barrier for the slice (every byte acked at every receiver), the pops,
 * and the flag: this member's word of every receiver's flags page set to the step's sequence by
 * a semaphore multicast from stage word ftnr_flag_stage_word(member) (same offset within 16
 * bytes as the destination word, or the write does not land), and this core's own word by a
 * plain store. A receiver therefore never sees the flag before the data. P0 sent faces 0-1 of
 * each page with a flushed barrier per page: 20 serial round trips a step, ~65 of its 100 us.
 *
 * THE OUTPUT. As resident_vector_writer.c (the replica map resolved once per step) but FACES
 * 0-1 of each tile only (kNrExchangeBytes), to every replica: the decode consumers of `normed`
 * read row-0 half tiles (kMatmulRow0), and faces 2-3 of the strip are the zero rows the shipped
 * norm also leaves zero. Each tile's writes go as it arrives with no barrier between tiles (Out
 * holds the whole slice), then one ack barrier and the pops before publishing.
 *
 * ROLE WORDS: kNrWordSteps (published steps), kNrWordBusy (cycles >> 4 from the reader's
 * precondition stamp to the publish) and kNrWordFlagSent (the same clock to this member's flag
 * sent) are this thread's alone.
 *
 * Compile-time args: 0 tile bytes (2048), 1 wait mode (1 stoppable, ships).
 * Runtime args: the ten of resident_queue_protocol.h.
 *
 * @status P0 (P0.2), exchange levers 1-2 (spec §11.11),
 *         faces / stagger modes (§11.13). Card gate pending for the modes.
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

/** Add the time from this step's precondition stamp to now to flags-page word `word` (the
 *  writer's own lines of the page, 28-31). `ctx->cursor` is this step's descriptor. */
static void flag_clock(ftq_context *ctx, uint32_t word)
{
    const volatile FTQ_L1_PTR uint32_t *stamps =
        (const volatile FTQ_L1_PTR uint32_t *)(uintptr_t)((uint32_t)(uintptr_t)ctx->control +
                                                          (uint32_t)kPreStampOffset);
    volatile FTQ_L1_PTR uint32_t *flags =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)ft_tensix_cb_base_pointer(kNrCbFlags);

    flags[word] +=
        (ft_bh_wall_clock_low() - stamps[ctx->cursor & ((uint32_t)kPreStampSlots - 1u)]) >> 4;
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

/** One run of `pages` Sq pages (ftnr_exchange_runs) to SqAll page member*S + tile on every
 *  member: per rectangle target, in the mode's order (ftnr_exchange_target), the mode's writes
 *  (one whole-page multicast, or one faces-0-1 multicast a page), then the whole run to this
 *  core in one write (local, so the bytes cost no link). No barrier here. */
static void send_run(const volatile FTQ_L1_PTR uint32_t *table, uint32_t member, uint32_t mode,
                     uint32_t source, uint32_t landing, uint32_t pages)
{
    const uint32_t targets = table[kNrHeadTargets];
    const uint32_t self_xy = table[kNrHeadSelfXy];
    const uint32_t writes = ftnr_exchange_writes(pages, mode);
    const uint32_t bytes = ftnr_exchange_write_bytes(pages, TILE_BYTES, mode);
    uint64_t address;

    for (uint32_t k = 0; k < targets; k++) {
        const uint32_t t = ftnr_exchange_target(k, member, targets, mode);

        if (target_receivers(table, t) == 0u || !target_address(table, t, landing, &address))
            continue;
        for (uint32_t w = 0; w < writes; w++)
            (void)ft_tensix_noc_write_multicast(source + w * TILE_BYTES,
                                                address + w * TILE_BYTES, bytes,
                                                target_receivers(table, t));
    }
    ft_tensix_noc_write(source, ft_tensix_noc_address(ftnr_x(self_xy), ftnr_y(self_xy), landing),
                        pages * TILE_BYTES);
}

/** This member's x^2 slice into every member's SqAll, then its flag: every run's writes, ONE
 *  write barrier for the slice, the pops, then the flag (spec §11.11 levers 1-2, §11.13). */
static bool send_exchange(ftq_context *ctx, const volatile FTQ_L1_PTR uint32_t *table,
                          volatile FTQ_L1_PTR uint32_t *flags, uint32_t member,
                          uint32_t sequence)
{
    const uint32_t slice = table[kNrHeadSliceTiles];
    const uint32_t targets = table[kNrHeadTargets];
    const uint32_t sqall = ft_tensix_cb_base_pointer(kNrCbSqAll);
    const uint32_t sq = ft_tensix_cb_base_pointer(kNrCbSq);
    const uint32_t flags_base = (uint32_t)(uintptr_t)flags;
    uint32_t first[kNrMaxExchangeRuns], tile[kNrMaxExchangeRuns], pages[kNrMaxExchangeRuns];
    uint64_t address;
    const bool have = ftq_cb_wait_front_stoppable(ctx, (int32_t)kNrCbSq, (int32_t)slice, WAIT_MODE);

    if (have && !ctx->aborting) {
        const uint32_t runs = ftnr_exchange_runs(
            (ft_tensix_cb_read_pointer(kNrCbSq) - sq) / TILE_BYTES,
            ft_tensix_cb(kNrCbSq)->fifo_num_pages, slice, first, tile, pages);

        if (runs == 0u) {
            ftq_abort(ctx, kErrBadParameters);
        } else {
            for (uint32_t r = 0; r < runs; r++)
                send_run(table, member, ftnr_exchange_mode(table), sq + first[r] * TILE_BYTES,
                         sqall + ftnr_sqall_offset(member, tile[r], slice, TILE_BYTES), pages[r]);
            ft_tensix_noc_write_barrier();
        }
    }
    ft_tensix_cb_pop_front((int32_t)kNrCbSq, (int32_t)slice);
    if (ctx->aborting)
        return false;
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

/** This member's normed slice to the output strip and its replicas: each tile's writes as it
 *  arrives, no barrier between tiles (Out holds the whole slice), then ONE write barrier and
 *  the pops (spec §11.11 lever 1 on the output). */
static bool write_output(ftq_context *ctx, const volatile FTQ_L1_PTR uint32_t *entry,
                         uint32_t member, uint32_t slice)
{
    const uint32_t xy = entry[kNrEntryOutXy];
    const uint32_t base = entry[kNrEntryOutBase];
    const uint32_t out = ft_tensix_cb_base_pointer(kNrCbOut);
    const uint32_t ring_pages = ft_tensix_cb(kNrCbOut)->fifo_num_pages;
    const uint32_t read_page = (ft_tensix_cb_read_pointer(kNrCbOut) - out) / TILE_BYTES;
    uint32_t destination_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t destination_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t destination_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    const uint32_t destinations =
        ftr_resolve_destinations((const volatile uint32_t *)(uintptr_t)FT_TT_REPLICA_TABLE_ADDRESS,
                                 ftnr_x(xy), ftnr_y(xy), base, destination_x, destination_y,
                                 destination_base);

    for (uint32_t i = 0; i < slice; i++) {
        const bool have = ftq_cb_wait_front_stoppable(ctx, (int32_t)kNrCbOut, (int32_t)(i + 1u),
                                                      WAIT_MODE);

        if (!have || ctx->aborting)
            break;
        if (i == 0u)
            flag_clock(ctx, kNrFlagOutFirst);
        const uint32_t source = out + ftnr_ring_page(read_page, i, ring_pages) * TILE_BYTES;
        const uint32_t offset = (member * slice + i) * TILE_BYTES;

        for (uint32_t copy = 0u; copy < destinations; copy++)
            ft_tensix_noc_write(source,
                                ft_tensix_noc_address(destination_x[copy], destination_y[copy],
                                                      destination_base[copy] + offset),
                                kNrExchangeBytes);
    }
    ft_tensix_noc_write_barrier();
    flag_clock(ctx, kNrFlagOutDone);
    ft_tensix_cb_pop_front((int32_t)kNrCbOut, (int32_t)slice);
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
