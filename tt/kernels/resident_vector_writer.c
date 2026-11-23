/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_vector_writer.c
 * @brief Writer third of BOTH resident vector triads -- elementwise and softmax -- and the
 * processor that PUBLISHES COMPLETION, per core and per group.
 *
 * ONE WRITER FOR THE WHOLE FAMILY, AND PROBABLY FOR MOST OF THE REMAINING OPS. A writer
 * is "drain `tile_count` tiles from CB_OUT to (x, y, base)". It does not know what
 * produced them, what shape they came from, or which function ran: those are the reader's
 * and the compute thread's business. Every op in resident_vector_protocol.h's family
 * therefore shares this source, and the only reason resident_matmul_writer.c is a
 * separate file is that the matmul parameter block spells the same three fields at
 * different word offsets. Unifying the two block layouts -- an "output descriptor"
 * sub-block every family shares -- is the follow-up that would make this ONE writer for
 * every op there will ever be. Worth doing before op 10, not after op 40.
 *
 * WHY THE WRITER OWNS COMPLETION (resident_queue_protocol.h rule 1). An op is not done
 * when the reader has fetched its operands or when the SFPU has finished; it is done when
 * the result has left L1. So this processor owns `completed_count` and `completed_token`,
 * the reader owns `tail`, and nothing else writes either.
 *
 * AND WHY IT FETCHES ON `head` BUT PUBLISHES ON `tail`. Fetching on `tail` is a pipeline
 * deadlock, not a precaution -- the output CB fills, compute blocks in cb_reserve_back,
 * the reader blocks because nothing drains the inputs, and the reader can never reach the
 * end of the descriptor to bump `tail`. resident_matmul_writer.c records the run that
 * cost. So the reader's half is a precondition of PUBLISHING and not of STARTING.
 *
 *
 * AND WHY IT FANS OUT. An activation strip lives in ONE of the card's eight ~59 GB/s DRAM
 * banks, and every member of a split matmul reads the WHOLE of it — so N members put N
 * readers on one bank and the card runs at 16% of its 511 GB/s peak however many cores it
 * spreads over. This writer is the producer of the two strips that carry ~88% of that
 * traffic (`normed`, a REDUCE output, and `ffn_swiglu`, an ELTWISE one), so it is the one
 * place a mirror can be made. What it costs is R NoC writes under ONE barrier; what it
 * buys is R banks' worth of read bandwidth on the consumer side. The map is
 * kernels/resident_output_replicas.h and it is EMPTY unless the host filled it in.
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_OUT index   1: output tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 * Descriptor words: a0 the parameter block; a3/a4 the S1 group pull on member 0 of a split
 *   descriptor (resident_group_pull.h); a5 the group sequence, stored in control word 56
 *   before completion is published. No reader or compute kernel of the family reads a1-a4.
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_vector_writer.c ncrisc /tmp/ttk_eltwise
 *
 * @status BUILT ON THE CARD HOST 2026-08-12, NOT RUN -- text 1936 B, bss 24 B on NCRISC. See
 *         tests/test_phase6_vector.c for the on-card acceptance.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_group_pull.h"
#include "resident_output_replicas.h"
#include "resident_queue_protocol.h"
#include "resident_vector_protocol.h"

#define CB_OUT ft_tensix_compile_time_arg(0)
#define OUT_TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
/**
 * Drain one descriptor's output tiles.
 *
 * The barrier is per tile, as it is in resident_matmul_writer.c and
 * vector_add_writer.c. That WAS a cost nobody had measured; it is now measured, and it
 * is large where nothing hides it: 546.9 ns a tile on MUL 144 -> DRAM, against 148.1 for
 * one barrier per 8 tiles and 178.8 for upstream's flush-per-tile shape
 * (resident_vector_writer_blocked.c and _flushed.c carry the tables). It is still
 * deliberately UNCHANGED here, because both cheaper arms are byte-identical measurement
 * arms nothing ships against, and because on an SFPU-bound op the whole difference
 * disappears — the SILU control moved 1.000x. Adopting one is a decision about which
 * invariant to carry, not about which number is smaller.
 */
static bool write_one_vector_op(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                                bool reduce)
{
    /* L4: a REDUCE descriptor with a column slice (kRedFinal byte 3, resident_reduce_protocol.h
     * ftr_slice) writes only tiles [first, first + count) of each row, at their whole-row
     * offsets. Decoded here without the reduce header: word 15, byte 3 = members<<4 | member. */
    const uint32_t tile_cols = parameters[kVecTileCols];
    const uint32_t slice_code = reduce ? (parameters[15] >> 24) & 0xFFu : 0u;
    const uint32_t slice_members = slice_code >> 4;
    const bool sliced = slice_members >= 2u && (slice_code & 0xFu) < slice_members &&
                        tile_cols != 0u && tile_cols % slice_members == 0u;
    const uint32_t slice_count = sliced ? tile_cols / slice_members : tile_cols;
    const uint32_t slice_first = sliced ? (slice_code & 0xFu) * slice_count : 0u;
    const uint32_t tile_count = sliced ? (parameters[kVecTileCount] / tile_cols) * slice_count
                                       : parameters[kVecTileCount];
    const uint32_t probe = ftv_probe_bits(parameters[kVecDepth]);
    uint32_t destination_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t destination_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t destination_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    /* THE REPLICA LOOKUP IS ONCE PER DESCRIPTOR, NOT ONCE PER TILE. The table is keyed on
     * the destination triple and cannot change under a descriptor, so hoisting it out of
     * the tile loop is free and keeps the unreplicated path exactly one indexed store and
     * one compare away from what shipped. An absent or empty table returns 1, and the loop
     * below is then the pre-replication loop instruction for instruction. */
    const uint32_t destinations =
        ftr_resolve_destinations((const volatile uint32_t *)(uintptr_t)FT_TT_REPLICA_TABLE_ADDRESS,
                                 parameters[kVecCX], parameters[kVecCY], parameters[kVecCBase],
                                 destination_x, destination_y, destination_base);

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        const bool have_tile = ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE);

        if (have_tile && (probe & kVecProbeNoWrite) == 0u) {
            const uint32_t source = ft_tensix_cb_read_pointer(CB_OUT);
            const uint32_t out_tile = sliced ? (tile / slice_count) * tile_cols + slice_first +
                                                   tile % slice_count
                                             : tile;
            const uint32_t offset = out_tile * OUT_TILE_BYTES;

            /* ONE BARRIER FOR ALL R WRITES, not R barriers. The barrier is what the 546.9
             * ns a tile is (148.1 at one per eight), so a per-replica barrier would make
             * the fan-out cost R times the whole write instead of R times the ISSUE. The
             * invariant the barrier carries is unchanged: the CB page must not be popped
             * until every read of it has landed, and that is a property of the SET of
             * writes, not of each one. */
            for (uint32_t copy = 0u; copy < destinations; copy++)
                ft_tensix_noc_write(source,
                                    ft_tensix_noc_address(destination_x[copy],
                                                          destination_y[copy],
                                                          destination_base[copy] + offset),
                                    OUT_TILE_BYTES);
            /* THE PAGE MAY BE POPPED ONCE ITS DATA HAS LEFT L1 (flushed) — all the pop
             * needs. The ACK barrier, which is what publishing needs, moves to the
             * descriptor's end (E3, 2026-10-04: the per-tile ack was ~14 us of a 33 us ADD;
             * decode +4.5%, IDENTICAL). kVecProbeLegacyBarrier restores the per-tile ack. */
            if ((probe & kVecProbeLegacyBarrier) != 0u)
                ft_tensix_noc_write_barrier();
            else
                ft_tensix_noc_writes_flushed_barrier();
        }
        ft_tensix_cb_pop_front(CB_OUT, 1);
    }
    /* Every write ACKED before the descriptor can be published (writes leave L1 at the
     * flush, but land in DRAM at the ack; the completion post must follow the landing). */
    ft_tensix_noc_write_barrier();
    return !ctx->aborting;
}

void kernel_main(void)
{
    ftq_context ctx;
    uint32_t completed;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    completed = ctx.control[kFieldCompletedCount];
    ctx.cursor = completed;

    while (running) {
        // FETCH on `head`, so the writer drains the output CB while the reader is still
        // filling the input ones. Publishing waits on `tail`, below.
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t group_sequence = descriptor[kDescA5];
        const bool vector_op = opcode == kOpEltwise || opcode == kOpSoftmax || opcode == kOpReduce;
        bool done = true;

        ftq_status_opcode(&ctx, opcode);
        /* kOpReduce carries the same three output words at the same offsets, which is the
         * whole reason one writer serves the family (file header). */
        if (vector_op) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = write_one_vector_op(&ctx, ftq_parameters(descriptor), opcode == kOpReduce);
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        // THE PRECONDITION, and the only place it belongs: the reader must have finished
        // this descriptor before its completion is published. For these ops that is already
        // true by construction — the last output tile cannot exist until the last input was
        // read — so it costs nothing here. For an opcode the writer does not participate
        // in, it is the whole of the guarantee.
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        /* S1 GROUP PULL (resident_group_pull.h): on member 0 of a split descriptor, a3 names
         * the members and a4 the sequence they must reach; it publishes only after every one
         * of them has. a3 = 0 (every unsplit descriptor) skips it. */
        {
            const uint32_t pull = descriptor[kDescA3];

            if (vector_op && (pull & ~kGroupPullFlags) > 1u &&
                (pull & kPullRedSkip) == 0u &&
                !ftq_group_pull(&ctx, pull & ~kGroupPullFlags, descriptor[kDescA4])) {
                running = false;
                continue;
            }
        }
        // The done-sequence a member 0 pull reads (control word 56), before the count.
        ctx.control[kFieldGroupDone] = group_sequence;
        // Token before count: a host that sees the count is entitled to trust the token it
        // reads next. Both are plain stores from ONE in-order RISC-V, which is the only
        // ordering argument this protocol makes and the only one it needs — every other
        // control word has a different single writer (ft_tt_queue.h, THE ORDERING
        // ASSUMPTION).
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    // The writer does NOT touch kFieldError or kFieldState: the reader owns those, and one
    // writer per word is the whole ordering argument.
    ftq_status_set(&ctx, kStateExit);
}
