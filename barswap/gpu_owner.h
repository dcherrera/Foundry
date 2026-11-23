/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_owner.h
 * @brief Multi-die GPU resource owner — public API + types (Phase 0 skeleton).
 *
 *   A single long-lived owner process that holds N K80 dies' CUDA contexts
 *   resident at once and serves work to any die by moving the one working MMIO
 *   window between dies — never tearing down.
 *
 *   Phase 0: types, the public API, and a command-dispatch REPL whose handlers
 *   are LOGGING STUBS. The real lifecycle (bring_die_online / swap_to / serve /
 *   ordered shutdown + the ordinal probe) lands in later phases.
 *
 * @status Phase-0 skeleton — REPL loop + lifecycle as logging stubs only.
 * @issues Real CUDA/barswap work is stubbed; no die ever actually comes online.
 * @todo
 *   - [ ] Phase 1: implement bring_die_online + the ordinal probe (spec §15.6)
 *   - [ ] Phase 2: implement swap_to (the hot path, no teardown)
 *   - [ ] Phase 3: implement serve (enqueue kernels, async collect)
 *   - [ ] Phase 4: implement ordered shutdown, replace _Exit() dodge (spec §11.1)
 */

#ifndef FOUNDRY_GPU_OWNER_H
#define FOUNDRY_GPU_OWNER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "barswap.h"
#include "gpu_owner_util.h"

/*
 * die_slot — resident per-die state. One live CUcontext per die, never
 * destroyed until owner shutdown. The MMIO window is a
 * movable viewport; every die keeps its full state resident across swaps.
 */
struct die_slot {
    int          die_id;
    char         bdf[16];        /* PCI address, e.g. "0000:03:00.0" */
    uint64_t     bar1_addr;      /* physical address of this die's VRAM window */
    CUcontext    ctx;            /* LIVE, never destroyed until owner shutdown */
    int          cuda_ordinal;   /* device ordinal nvidia assigned this die */
    CUdeviceptr  d_in;           /* resident input buffer handle */
    CUdeviceptr  d_out;          /* resident output buffer handle */
    size_t       buf_bytes;      /* size of the resident buffers */
    int          bound;          /* nvidia bound this die's PCI function? */
    int          online;         /* has this die ever been brought online? */
    int          ctx_valid;      /* last known-good? */
    uint64_t     last_used_ns;
    CUmodule     mod;            /* smoke-kernel module loaded into this ctx */
    CUfunction   kern;           /* "slow_add" entry within mod */
    uint32_t     last_seed;      /* seed of the last work() — for read() verify */
    int          last_iters;     /* iters of the last work() — for read() verify */
};

/*
 * gpu_owner — the owner state. Opaque to callers (full definition in the .c).
 * Holds the barswap context, the dlopen'd libcuda symbol table, and the
 * resident per-die slot table.
 */
typedef struct gpu_owner gpu_owner;

/*
 * REPL command grammar (one command per line, whitespace-separated; Phase-0
 * handlers LOG what they WOULD do, prefixed "[stub]", and update only trivial
 * harmless state such as active_die):
 *
 *   discover                 enumerate the 16 GK210 dies; find the slot occupant
 *   online <die>             bring a die online (acquire, bind, create ctx, upload)
 *   swap <die>               move the MMIO window to <die>, set its ctx current
 *   work <die> <seed> <iters>  enqueue the smoke kernel on <die> with a seed pattern
 *   read <die>               copy <die>'s output buffer back and verify the pattern
 *   count                    report cuDeviceGetCount (the ordinal probe, spec §15.6)
 *   status                   print the per-die slot table + active die
 *   help                     print this grammar
 *   shutdown | quit          ordered teardown (Phase 0: stub + exit the REPL)
 */

/*
 * Construct the owner: open barswap and load the libcuda symbol table.
 *
 * @return  Heap-allocated owner on success; NULL if barswap is absent (Mac /
 *          no K80 → the caller's SKIP path) or on allocation failure.
 *
 * Ownership: caller frees via gpu_owner_destroy().
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
gpu_owner *gpu_owner_create(void);

/*
 * Destroy the owner and free its resources.
 *
 * Phase 0: frees the struct + barswap context only (no live CUDA state exists
 * yet). Phase 4 adds the ordered per-die teardown (spec §11.1). NULL-safe.
 */
void gpu_owner_destroy(gpu_owner *owner);

/*
 * Run the command-dispatch REPL until "shutdown"/"quit" or EOF.
 *
 * @param owner      The owner (must be non-NULL).
 * @param fifo_path  Optional path to a command FIFO; NULL → read from stdin.
 * @param log_path   Optional path to a log file; NULL → log to stdout.
 * @return           0 on clean exit, negative on error.
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
int gpu_owner_run_repl(gpu_owner *owner, const char *fifo_path,
                       const char *log_path);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_GPU_OWNER_H */
