/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_owner.c (tests/)
 * @brief Runnable entry point for the multi-die GPU resource owner.
 *
 *   Constructs the owner and runs its command-dispatch REPL. On the Mac / off
 *   a K80 host — no /dev/bar_swap and no libcuda — it prints [SKIP] and exits
 *   0 so the build's test sweep stays green everywhere. On a K80 host it drives
 *   the owner from stdin by default; optional argv gives a FIFO + log path.
 *
 *   Usage: gpu_owner [fifo_path [log_path]]
 *
 * @status Phase-0 — drives the stub REPL. Teardown is the _Exit(0) dodge.
 * @issues _Exit(0) skips the ordered shutdown — fine while no real CUDA context
 *   exists; Phase 4 replaces it with gpu_owner ordered teardown (spec §11.1).
 * @todo
 *   - [ ] Phase 4: swap _Exit(0) for owner ordered shutdown (spec §11.1)
 */

/* 4. Project headers */
#include "barswap/gpu_owner.h"
#include "barswap/gpu_owner_util.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    gpu_owner *owner = NULL;
    const char *fifo_path = argc > 1 ? argv[1] : NULL;
    const char *log_path = argc > 2 ? argv[2] : NULL;

    /* The Mac / no-K80 SKIP path: gpu_owner_create() returns NULL when
     * /dev/bar_swap is absent (bs_init NULL), and gou_cuda_load_symbols()
     * returns NULL when there's no libcuda. Either one → not on a K80 host. */
    owner = gpu_owner_create();
    if (!owner || !gou_cuda_load_symbols()) {
        printf("[SKIP] no barswap/CUDA — not on a K80 host\n");
        gpu_owner_destroy(owner);
        return 0;
    }

    gpu_owner_run_repl(owner, fifo_path, log_path);

    /* The owner NEVER tears down: no bs_release, no
     * cuCtxDestroy, no bs_free while dies are live. CRITICAL: do NOT release the
     * bar_swap slot on exit. bar_swap has no fd-close handler (bs_fops is only
     * .unlocked_ioctl), so leaving the slot mapped means nvidia's own fd-close
     * context teardown happens with the BAR still present — safe. Releasing the
     * slot first yanks the BAR from under the driver and the GPU "falls off the
     * bus" (Xid 79, observed 2026-06-27 from a self-inflicted exit-time
     * bs_release, since removed). So: leak everything by design and let the
     * kernel reclaim the process. */
    _Exit(0);
}
