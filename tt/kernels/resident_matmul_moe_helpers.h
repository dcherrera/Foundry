/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_moe_helpers.h
 * @brief The BFP6_B expansion split over the five RISC-Vs of one `rtriad_bfp6` core (qwen36
 *   plan S3, spec §4.1b route 3): BRISC posts each landed block as a job in the mailbox
 *   (resident_matmul_moe_protocol.h), every participating processor expands its share of the
 *   block's tiles out of place into the CB_IN1 pages, and BRISC pushes the pages once every
 *   helper's done word names the job.
 *
 * ONE JOB A BLOCK, COUNTED BY EVERYONE. The reader posts a job for EVERY block of every
 * descriptor it runs (a drained block posts zero tiles), and the compute threads and the
 * writer call ftmoe_help() once per block, walking the same block order from the same
 * descriptor words. So the k-th job is the same block on all five processors and no id is
 * ever negotiated. The host zeroes the mailbox at every arm and each processor zeroes its
 * count at boot (ftmoe_boot()), so ids restart with the residency.
 *
 * WHY THIS IS SAFE FOR THE PIPELINE. A helper writes only the pages of the block whose job
 * it holds, and BRISC does not push those pages until the helper's done word names that
 * job. The compute threads help with block b before they consume it: MATH and PACK run
 * ahead of the data in software (their waits are in the Tensix backend), so they reach
 * block b's job after issuing block b-1, which was pushed already. No wait here can close
 * a cycle: every job depends only on blocks pushed before it.
 *
 * Every wait is stoppable and bounded like a CB wait (a stop request or the spin bound
 * aborts the descriptor); a processor that aborted skips its share, and the residency then
 * exits as on any abort.
 *
 * Defines: FT_RTRIAD_MOE_WORKERS (1..5, default 5) processors share a block, worker w being
 *   0 BRISC, 1 NCRISC, 2 TRISC0, 3 TRISC1, 4 TRISC2; a processor >= W only counts jobs.
 *   FT_RTRIAD_MOE_RED_SHIFT=1 the red expander (off by one bit).
 *
 * @status New 2026-11-02 (qwen36 plan S3).
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_MATMUL_MOE_HELPERS_H
#define RESIDENT_MATMUL_MOE_HELPERS_H

#include <stdbool.h>
#include <stdint.h>

#include "bfp6_expand_tile.h"
#include "resident_matmul_moe_protocol.h"
#include "resident_queue_protocol.h"

#ifndef FT_RTRIAD_MOE_WORKERS
#define FT_RTRIAD_MOE_WORKERS 5u
#endif
#if FT_RTRIAD_MOE_WORKERS < 1 || FT_RTRIAD_MOE_WORKERS > 5
#error "FT_RTRIAD_MOE_WORKERS must be 1..5"
#endif
#if defined(FT_RTRIAD_MOE_RED_SHIFT) && FT_RTRIAD_MOE_RED_SHIFT
#define FTMOE_EXPAND_SHIFT 1u
#else
#define FTMOE_EXPAND_SHIFT 2u
#endif

/** Jobs this processor has seen since its kernel booted. */
static uint32_t ftmoe_jobs;

static inline void ftmoe_boot(void)
{
    ftmoe_jobs = 0u;
}

static inline volatile FTQ_L1_PTR uint32_t *ftmoe_mailbox(const ftq_context *ctx)
{
    return (volatile FTQ_L1_PTR uint32_t *)((uintptr_t)ctx->control + FT_TT_MOE_MAILBOX_OFFSET);
}

/** Worker `worker`'s share [first, end) of `tiles`. */
static inline void ftmoe_share(uint32_t tiles, uint32_t worker, uint32_t *first, uint32_t *end)
{
    *first = tiles * worker / FT_RTRIAD_MOE_WORKERS;
    *end = tiles * (worker + 1u) / FT_RTRIAD_MOE_WORKERS;
}

/** Zero-fill worker `worker`'s share of a job (source 0: a LIST-mode entry past the list, an
 *  all-zero weight block, so its products are exact zeros). Its OWN non-inlined function: the
 *  same loop inside ftmoe_expand_share cost the expansion 0.309 -> 0.398 us a weight tile
 *  (run_q36_s1_price_bisect.sh, 2026-11-02), a code-generation effect on the hot loop. */
static __attribute__((noinline)) void ftmoe_zero_share(uint32_t target, uint32_t tiles,
                                                       uint32_t worker)
{
    uint32_t first, end;

    ftmoe_share(tiles, worker, &first, &end);
    volatile uint32_t *out = (volatile uint32_t *)(uintptr_t)(target + first * BFP8_TILE_BYTES);

    for (uint32_t w = 0; w < (end - first) * (BFP8_TILE_BYTES / 4u); w++)
        out[w] = 0u;
}

/** Expand worker `worker`'s share of a job: tile t from `source + t * 832` (FOLD layout) to
 *  `target + t * 1088`. Keep this function exactly as it is (see ftmoe_zero_share). */
static inline void ftmoe_expand_share(uint32_t source, uint32_t target, uint32_t tiles,
                                      uint32_t worker)
{
    uint32_t first, end;

    ftmoe_share(tiles, worker, &first, &end);
    for (uint32_t t = first; t < end; t++)
        bfp6_expand_one_tile((const volatile uint32_t *)(uintptr_t)(source + t * BFP6_TILE_BYTES),
                             (volatile uint32_t *)(uintptr_t)(target + t * BFP8_TILE_BYTES),
                             BFP6_LAYOUT_FOLD, FTMOE_EXPAND_SHIFT);
}

/** Spin on `word == want` with a CB wait's escapes. false = stopped or timed out (aborted). */
static inline bool ftmoe_wait_word(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *word,
                                   uint32_t want)
{
    uint32_t spins = 0u;

    for (;;) {
        ft_bh_l1_cache_invalidate();
        if (*word == want)
            return true;
        if ((++spins & (uint32_t)kStopPollMask) == 0u) {
            if (ftq_stop_requested(ctx)) {
                ftq_abort(ctx, kErrNone);
                return false;
            }
            if (spins >= ctx->cb_spin_limit) {
                ftq_abort(ctx, kErrCbWaitTimeout);
                return false;
            }
        }
    }
}

/**
 * HELPER side (workers 1..4): take the next job, expand this worker's share, publish done.
 * A processor that does not take part (worker >= W) only counts the job.
 * @return false if the descriptor is aborting (the share was skipped)
 */
static inline bool ftmoe_help(ftq_context *ctx, uint32_t worker)
{
    const uint32_t id = ++ftmoe_jobs;
    volatile FTQ_L1_PTR uint32_t *box;

    if (worker >= FT_RTRIAD_MOE_WORKERS)
        return true;
    if (ctx->aborting)
        return false;
    box = ftmoe_mailbox(ctx);
    if (!ftmoe_wait_word(ctx, &box[kMoeJob], id))
        return false;
    {
        const uint32_t source = box[kMoeJobSource];

        if (source == 0u)
            ftmoe_zero_share(box[kMoeJobTarget], box[kMoeJobTiles], worker);
        else
            ftmoe_expand_share(source, box[kMoeJobTarget], box[kMoeJobTiles], worker);
    }
    ft_bh_l1_cache_invalidate(); /* the share's stores land before the done word */
    box[kMoeDone1 + worker - 1u] = id;
    return true;
}

/**
 * BRISC side: post the block landed at `source` as the next job for `target`, expand
 * BRISC's share, then wait for every helper. `tiles` = 0 posts a drained block (helpers
 * only count it).
 * @return false if a wait was abandoned (the descriptor is aborting)
 */
static inline bool ftmoe_expand_job(ftq_context *ctx, uint32_t source, uint32_t target,
                                    uint32_t tiles)
{
    const uint32_t id = ++ftmoe_jobs;
    volatile FTQ_L1_PTR uint32_t *box = ftmoe_mailbox(ctx);
    bool ok = true;

    if (FT_RTRIAD_MOE_WORKERS > 1u) {
        box[kMoeJobSource] = source;
        box[kMoeJobTarget] = target;
        box[kMoeJobTiles] = tiles;
        ft_bh_l1_cache_invalidate(); /* the job's words land before its id */
        box[kMoeJob] = id;
    }
    ft_bh_l1_cache_invalidate(); /* the staged block, not a stale cached copy */
    if (source == 0u)
        ftmoe_zero_share(target, tiles, 0u);
    else
        ftmoe_expand_share(source, target, tiles, 0u);
    for (uint32_t w = 1u; w < FT_RTRIAD_MOE_WORKERS && ok; w++)
        ok = ftmoe_wait_word(ctx, &box[kMoeDone1 + w - 1u], id);
    ft_bh_l1_cache_invalidate();
    return ok;
}

#endif /* RESIDENT_MATMUL_MOE_HELPERS_H */
