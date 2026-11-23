/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gpu_owner.c
 * @brief Implementation of the multi-die GPU resource owner.
 *
 *   The owner library: lifecycle functions + a command-dispatch REPL. A single
 *   long-lived process holds per-die CUDA contexts resident and serves work by
 *   moving the one working MMIO window between dies.
 *
 *   Phase 1 (this file): the real SINGLE-DIE lifecycle — bring_die_online
 *   (acquire → block_all_except → load nvidia once → cuInit/cuCtxCreate → load
 *   the smoke kernel → alloc resident buffers), work (fill seed + launch), read
 *   (sync + copy back + verify), and the ordinal probe (cuDeviceGetCount). The
 *   second-die bring-up (the crash crux) and the no-teardown hot-path swap land
 *   in Phase 2; ordered shutdown lands in Phase 4.
 *
 *   Logging is loud, unbuffered, and timestamped so a host hang names the exact
 *   line that wedged (spec §11.7).
 *
 * @status Phase-1 — single-die lifecycle real; multi-die swap stubbed.
 * @issues By design the owner NEVER tears down: exit is _Exit() with the slot
 *   left mapped, so nvidia's own fd-close teardown keeps the BAR present.
 *   Releasing the slot before exit pulls the BAR and the GPU falls off the bus
 *   (Xid 79, observed 2026-06-27). Safe multi-die stop (N contexts, one slot,
 *   no cuCtxDestroy) is still open.
 * @todo
 *   - [ ] Phase 2: second-die bring-up + swap_to across two contexts (spec §6)
 *   - [ ] Multi-die safe stop without cuCtxDestroy (open; spec §11.1)
 */

/* 1. This module's own header */
#include "gpu_owner.h"

/* 2. Standard library */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3. System / platform headers */
#include <time.h>

/* 4. Project headers */
#include "barswap.h"
#include "gpu_owner_util.h"

/* Smoke-kernel payload: 1M floats = 4 MB per buffer, 256 threads/block. The
 * slow_add kernel computes out[i] = in[i] + iters (exact in fp32 under 2^24). */
#define OWNER_N_ELEMENTS  (1024 * 1024)
#define OWNER_BLOCK_SIZE  256

/*
 * gpu_owner — full definition (opaque in the header). Mirrors spec §5 plus the
 * Phase-1 process-wide handles: the dlopen'd CUDA table, the single mapped-die
 * handle, and the once-only init latches.
 */
struct gpu_owner {
    bs_context     *bs;                    /* the "move the one MMIO window" primitive */
    const gou_cuda *cu;                    /* dlopen'd libcuda symbol table */
    struct die_slot dies[GOU_MAX_DIES];    /* resident per-die slot table */
    bs_gpu         *active_gpu;            /* handle for the die currently in the slot */
    int             active_die;            /* die currently in the MMIO slot, or -1 */
    int             nvidia_loaded;         /* has nvidia been modprobe'd this run? */
    int             cuda_inited;           /* has cuInit() run this run? */
    int             discovered;            /* has gou_discover_gk210s() run? */
    FILE           *log;                   /* log sink (stdout unless a path is given) */
};

/* ------------------------------------------------------------------------- */
/* Logging — timestamped, unbuffered, grep-friendly (spec §11.7).            */
/* ------------------------------------------------------------------------- */

/*
 * Emit a timestamped log line to the owner's sink. Always one line, flushed
 * immediately so a hang is visible at the exact step instead of a silent stall.
 */
static void owner_log(const gpu_owner *owner, const char *fmt, ...)
{
    struct timespec ts = {0};
    struct tm tm_now = {0};
    char stamp[32] = {0};
    FILE *sink = owner && owner->log ? owner->log : stdout;
    va_list args;

    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm_now);
    strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm_now);

    fprintf(sink, "[%s.%03ld] ", stamp, ts.tv_nsec / 1000000L);
    va_start(args, fmt);
    vfprintf(sink, fmt, args);
    va_end(args);
    fputc('\n', sink);
    fflush(sink);
}

/*
 * Check a CUDA driver-API result; log and return 0 on failure, 1 on success.
 * Keeps the bring-up path readable while still naming the failing call.
 */
static int cu_ok(const gpu_owner *owner, CUresult rc, const char *what)
{
    if (rc != 0) {
        owner_log(owner, "  CUDA %s FAILED rc=%d", what, rc);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Lifecycle.                                                                */
/* ------------------------------------------------------------------------- */

gpu_owner *gpu_owner_create(void)
{
    gpu_owner *owner = NULL;

    owner = calloc(1, sizeof(*owner));
    if (!owner)
        return NULL;

    owner->log = stdout;
    owner->active_die = -1;
    for (int i = 0; i < GOU_MAX_DIES; i++) {
        owner->dies[i].die_id = i;
        owner->dies[i].cuda_ordinal = -1;
    }

    /* bs_init returns NULL when /dev/bar_swap is absent (Mac / no K80). The
     * caller's main() also checks gou_cuda_load_symbols() for the SKIP path. */
    owner->bs = bs_init();
    if (!owner->bs) {
        free(owner);
        return NULL;
    }

    owner->cu = gou_cuda_load_symbols();

    return owner;
}

void gpu_owner_destroy(gpu_owner *owner)
{
    if (!owner)
        return;

    /* The owner NEVER releases a live die or destroys a context — that is the
     * architecture (never tear down, multiDie_spec.md §3). bs_free only closes
     * the /dev/bar_swap fd; it does NOT release the slot (bar_swap has no
     * fd-close handler), so the mapped die stays mapped for nvidia's own
     * teardown. Used only on the SKIP / no-die-acquired path; the live owner
     * _Exit()s without ever calling this. */
    bs_free(owner->bs);
    owner->bs = NULL;
    free(owner);
}

/* ------------------------------------------------------------------------- */
/* Real single-die operations (Phase 1).                                     */
/* ------------------------------------------------------------------------- */

/*
 * Move the single MMIO window onto `die`: release the currently-mapped handle
 * (munmap + RELEASE) then bs_acquire the target. No-op if already mapped. This
 * is the swap primitive; Phase 2 stresses it across two coexisting contexts.
 */
static int owner_acquire_die(gpu_owner *owner, int die)
{
    bs_gpu *g = NULL;

    if (owner->active_die == die && owner->active_gpu)
        return 0;

    if (owner->active_gpu) {
        bs_release(owner->bs, owner->active_gpu);
        owner->active_gpu = NULL;
        owner->active_die = -1;
    }

    g = bs_acquire(owner->bs, die);
    if (!g) {
        owner_log(owner, "  bs_acquire(%d) FAILED", die);
        return -1;
    }
    owner->active_gpu = g;
    owner->active_die = die;
    owner->dies[die].bar1_addr = g->bar1_addr;
    owner_log(owner, "  die %d in MMIO slot (bar1=0x%llx)", die,
              (unsigned long long)g->bar1_addr);
    return 0;
}

/*
 * Bring one die fully online: into the slot, nvidia bound (first time only,
 * blocking all other GK210s), a live CUDA context, the smoke kernel loaded, and
 * resident in/out buffers allocated. Single-die: device ordinal is 0 (only one
 * GK210 is bound). spec §5 bring_die_online.
 */
static int owner_bring_die_online(gpu_owner *owner, int die)
{
    struct die_slot *d = &owner->dies[die];
    CUdevice dev = 0;
    int idx = -1;

    if (!owner->cu) {
        owner_log(owner, "online %d: no CUDA table (no K80)", die);
        return -1;
    }
    if (!owner->discovered) {
        owner_log(owner, "online %d: discovered %d GK210 dies", die,
                  gou_discover_gk210s());
        owner->discovered = 1;
    }
    if (owner_acquire_die(owner, die) != 0)
        return -1;

    idx = gou_find_acquired_bdf();
    if (idx >= 0)
        snprintf(d->bdf, sizeof(d->bdf), "%s", gou_gk210_bdf(idx));
    owner_log(owner, "  acquired bdf index=%d (%s)", idx, d->bdf[0] ? d->bdf : "?");

    if (!owner->nvidia_loaded) {
        gou_block_all_except(idx);
        owner_log(owner, "  blocked all GK210s except idx=%d; loading nvidia...", idx);
        if (gou_load_nvidia_once() != 0) {
            owner_log(owner, "  load_nvidia_once FAILED");
            return -1;
        }
        owner->nvidia_loaded = 1;
        owner_log(owner, "  nvidia loaded");
    }
    if (!owner->cuda_inited) {
        if (!cu_ok(owner, owner->cu->cuInit(0), "cuInit"))
            return -1;
        owner->cuda_inited = 1;
    }
    if (!cu_ok(owner, owner->cu->cuDeviceGet(&dev, 0), "cuDeviceGet(0)"))
        return -1;
    if (!cu_ok(owner, owner->cu->cuCtxCreate(&d->ctx, 0, dev), "cuCtxCreate"))
        return -1;
    d->cuda_ordinal = 0;
    if (!cu_ok(owner, owner->cu->cuModuleLoadData(&d->mod, gou_slow_add_ptx),
               "cuModuleLoadData") ||
        !cu_ok(owner, owner->cu->cuModuleGetFunction(&d->kern, d->mod, "slow_add"),
               "cuModuleGetFunction"))
        return -1;

    d->buf_bytes = (size_t)OWNER_N_ELEMENTS * sizeof(float);
    if (!cu_ok(owner, owner->cu->cuMemAlloc(&d->d_in, d->buf_bytes), "cuMemAlloc(in)") ||
        !cu_ok(owner, owner->cu->cuMemAlloc(&d->d_out, d->buf_bytes), "cuMemAlloc(out)"))
        return -1;

    d->online = 1;
    d->ctx_valid = 1;
    owner_log(owner, "online %d: OK (ctx=%p ordinal=%d in=0x%llx out=0x%llx)",
              die, d->ctx, d->cuda_ordinal,
              (unsigned long long)d->d_in, (unsigned long long)d->d_out);
    return 0;
}

/*
 * Fill die's input buffer with the seed pattern and launch the smoke kernel
 * (out = in + iters) on its context. Does NOT sync — read() collects. The seed
 * + iters are recorded so read() can verify without re-passing them.
 */
static int owner_work(gpu_owner *owner, int die, uint32_t seed, int iters)
{
    struct die_slot *d = &owner->dies[die];
    float *host = NULL;
    unsigned int n = OWNER_N_ELEMENTS;
    unsigned int it = (unsigned int)iters;
    unsigned int grid = (n + OWNER_BLOCK_SIZE - 1) / OWNER_BLOCK_SIZE;
    void *params[4];
    int rc = -1;

    if (!d->online) {
        owner_log(owner, "work %d: die not online — run 'online %d' first", die, die);
        return -1;
    }
    if (owner_acquire_die(owner, die) != 0)
        return -1;
    if (!cu_ok(owner, owner->cu->cuCtxSetCurrent(d->ctx), "cuCtxSetCurrent"))
        return -1;

    host = malloc(d->buf_bytes);
    if (!host)
        return -1;
    gou_fill_die_pattern(host, OWNER_N_ELEMENTS, seed);
    if (!cu_ok(owner, owner->cu->cuMemcpyHtoD(d->d_in, host, d->buf_bytes),
               "cuMemcpyHtoD"))
        goto cleanup;

    params[0] = &d->d_out;
    params[1] = &d->d_in;
    params[2] = &n;
    params[3] = &it;
    if (!cu_ok(owner, owner->cu->cuLaunchKernel(d->kern, grid, 1, 1,
                                                OWNER_BLOCK_SIZE, 1, 1, 0, NULL,
                                                params, NULL), "cuLaunchKernel"))
        goto cleanup;

    d->last_seed = seed;
    d->last_iters = iters;
    owner_log(owner, "work %d: launched (seed=0x%08x iters=%d grid=%u) — not synced",
              die, seed, iters, grid);
    rc = 0;
cleanup:
    free(host);
    return rc;
}

/*
 * Swap to die, sync, copy its output buffer back, and verify out = in + iters
 * against the recorded seed (the cheap per-die context validator, spec §7).
 */
static int owner_read(gpu_owner *owner, int die)
{
    struct die_slot *d = &owner->dies[die];
    float *host = NULL;
    int ok = 0;

    if (!d->online) {
        owner_log(owner, "read %d: die not online", die);
        return -1;
    }
    if (owner_acquire_die(owner, die) != 0)
        return -1;
    if (!cu_ok(owner, owner->cu->cuCtxSetCurrent(d->ctx), "cuCtxSetCurrent") ||
        !cu_ok(owner, owner->cu->cuCtxSynchronize(), "cuCtxSynchronize"))
        return -1;

    host = malloc(d->buf_bytes);
    if (!host)
        return -1;
    if (cu_ok(owner, owner->cu->cuMemcpyDtoH(host, d->d_out, d->buf_bytes),
              "cuMemcpyDtoH")) {
        ok = gou_verify_die_pattern(host, OWNER_N_ELEMENTS, d->last_seed,
                                    d->last_iters);
        owner_log(owner, "read %d: %s (sample out[0]=%.1f out[%d]=%.1f, "
                         "expect in+%d)", die, ok ? "PASS" : "FAIL", host[0],
                  OWNER_N_ELEMENTS - 1, host[OWNER_N_ELEMENTS - 1], d->last_iters);
    }
    free(host);
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------------- */
/* Per-command handlers.                                                     */
/* ------------------------------------------------------------------------- */

static void handle_discover(gpu_owner *owner)
{
    int n = gou_discover_gk210s();
    int slot = gou_find_acquired_bdf();
    owner->discovered = 1;
    owner_log(owner, "discover: %d GK210 dies; slot occupant idx=%d (%s)", n, slot,
              slot >= 0 ? gou_gk210_bdf(slot) : "none");
    for (int i = 0; i < n; i++)
        owner_log(owner, "  gk210[%d] = %s", i, gou_gk210_bdf(i));
}

static void handle_online(gpu_owner *owner, int die)
{
    if (die < 0 || die >= GOU_MAX_DIES) {
        owner_log(owner, "online: bad die %d (range 0..%d)", die, GOU_MAX_DIES - 1);
        return;
    }
    owner_bring_die_online(owner, die);
}

static void handle_swap(gpu_owner *owner, int die)
{
    if (die < 0 || die >= GOU_MAX_DIES) {
        owner_log(owner, "swap: bad die %d (range 0..%d)", die, GOU_MAX_DIES - 1);
        return;
    }
    /* Phase 1: single-die swap = the acquire primitive. Phase 2 stresses this
     * across two coexisting contexts (the crash crux, spec §6/§11.8). */
    if (owner_acquire_die(owner, die) == 0 && owner->dies[die].ctx)
        owner->cu->cuCtxSetCurrent(owner->dies[die].ctx);
}

static void handle_work(gpu_owner *owner, int die, uint32_t seed, int iters)
{
    if (die < 0 || die >= GOU_MAX_DIES) {
        owner_log(owner, "work: bad die %d (range 0..%d)", die, GOU_MAX_DIES - 1);
        return;
    }
    owner_work(owner, die, seed, iters);
}

static void handle_read(gpu_owner *owner, int die)
{
    if (die < 0 || die >= GOU_MAX_DIES) {
        owner_log(owner, "read: bad die %d (range 0..%d)", die, GOU_MAX_DIES - 1);
        return;
    }
    owner_read(owner, die);
}

static void handle_count(gpu_owner *owner)
{
    int cnt = -1;
    /* The ordinal probe (spec §6.2 / §15.6): with N dies bound, N → per-die
     * ordinals; 1 → the single in-slot die is always ordinal 0. */
    if (!owner->cu) {
        owner_log(owner, "count: no CUDA table (no K80)");
        return;
    }
    cu_ok(owner, owner->cu->cuDeviceGetCount(&cnt), "cuDeviceGetCount");
    owner_log(owner, "count: cuDeviceGetCount = %d (N=per-die ordinals, "
                     "1=always-ordinal-0)", cnt);
}

/*
 * The Phase-2 crux probe (spec §6, §11.8): bring a SECOND die into the slot and
 * bind it to nvidia WHILE the first die stays bound — the exact step that
 * crash-rebooted the box under worker_switch. Isolated: acquire → unblock →
 * sysfs-bind → cuDeviceGetCount. NO context is created here. Each sub-step is
 * logged before it runs so a wedge/crash names the precise line (read dmesg).
 */
static void handle_probe(gpu_owner *owner, int die)
{
    struct die_slot *d = &owner->dies[die];
    char val[40] = {0};
    int idx = -1;
    int cnt = -1;
    int rc = -1;

    if (die < 0 || die >= GOU_MAX_DIES) {
        owner_log(owner, "probe: bad die %d (range 0..%d)", die, GOU_MAX_DIES - 1);
        return;
    }
    if (!owner->cu || !owner->nvidia_loaded) {
        owner_log(owner, "probe %d: bring a first die online first (need nvidia "
                         "loaded)", die);
        return;
    }
    owner_log(owner, "probe %d: acquiring into slot (releases die %d — stays "
                     "resident)...", die, owner->active_die);
    if (owner_acquire_die(owner, die) != 0)
        return;
    idx = gou_find_acquired_bdf();
    if (idx >= 0)
        snprintf(d->bdf, sizeof(d->bdf), "%s", gou_gk210_bdf(idx));
    owner_log(owner, "probe %d: in slot as %s (idx=%d); unblocking override...",
              die, d->bdf[0] ? d->bdf : "?", idx);
    gou_set_driver_override(d->bdf, "");

    snprintf(val, sizeof(val), "%s\n", d->bdf);
    owner_log(owner, "probe %d: >>> BINDING %s to nvidia (THE crux step) <<<",
              die, d->bdf);
    rc = gou_sysfs_write("/sys/bus/pci/drivers/nvidia/bind", val);
    owner_log(owner, "probe %d: bind returned rc=%d (survived the bind)", die, rc);

    cu_ok(owner, owner->cu->cuDeviceGetCount(&cnt), "cuDeviceGetCount");
    owner_log(owner, "probe %d: cuDeviceGetCount = %d  (was 1 with one die; "
                     "2 => per-die ordinals, the §13-Q2 answer)", die, cnt);
}

static void handle_status(gpu_owner *owner)
{
    owner_log(owner, "status: active_die=%d nvidia_loaded=%d cuda_inited=%d",
              owner->active_die, owner->nvidia_loaded, owner->cuda_inited);
    for (int i = 0; i < GOU_MAX_DIES; i++) {
        const struct die_slot *d = &owner->dies[i];
        if (!d->online)
            continue;
        owner_log(owner, "  die %2d: online=%d bound=%d ctx_valid=%d ordinal=%d "
                         "bdf=%s", d->die_id, d->online, d->bound, d->ctx_valid,
                  d->cuda_ordinal, d->bdf[0] ? d->bdf : "?");
    }
}

static void handle_help(gpu_owner *owner)
{
    owner_log(owner, "commands: discover | online <die> | probe <die> | "
                     "swap <die> | work <die> <seed> <iters> | read <die> | "
                     "count | status | help | shutdown|quit");
}

/* ------------------------------------------------------------------------- */
/* Command dispatch + the REPL loop.                                         */
/* ------------------------------------------------------------------------- */

/*
 * Dispatch one parsed command line. Returns 1 to keep looping, 0 to stop
 * (shutdown/quit). Unknown commands log a hint and keep looping.
 */
static int dispatch_command(gpu_owner *owner, char *line)
{
    char *cmd = strtok(line, " \t\r\n");

    if (!cmd)
        return 1;

    if (strcmp(cmd, "discover") == 0) {
        handle_discover(owner);
    } else if (strcmp(cmd, "online") == 0) {
        const char *a = strtok(NULL, " \t\r\n");
        handle_online(owner, a ? atoi(a) : -1);
    } else if (strcmp(cmd, "swap") == 0) {
        const char *a = strtok(NULL, " \t\r\n");
        handle_swap(owner, a ? atoi(a) : -1);
    } else if (strcmp(cmd, "work") == 0) {
        const char *a = strtok(NULL, " \t\r\n");
        const char *b = strtok(NULL, " \t\r\n");
        const char *c = strtok(NULL, " \t\r\n");
        handle_work(owner, a ? atoi(a) : -1,
                    b ? (uint32_t)strtoul(b, NULL, 0) : 0u, c ? atoi(c) : 0);
    } else if (strcmp(cmd, "read") == 0) {
        const char *a = strtok(NULL, " \t\r\n");
        handle_read(owner, a ? atoi(a) : -1);
    } else if (strcmp(cmd, "count") == 0) {
        handle_count(owner);
    } else if (strcmp(cmd, "probe") == 0) {
        const char *a = strtok(NULL, " \t\r\n");
        handle_probe(owner, a ? atoi(a) : -1);
    } else if (strcmp(cmd, "status") == 0) {
        handle_status(owner);
    } else if (strcmp(cmd, "help") == 0) {
        handle_help(owner);
    } else if (strcmp(cmd, "shutdown") == 0 || strcmp(cmd, "quit") == 0) {
        owner_log(owner, "%s: leaving REPL — NO teardown (dies stay resident, "
                         "slot stays mapped); main() _Exit's", cmd);
        return 0;
    } else {
        owner_log(owner, "unknown command '%s' — try 'help'", cmd);
    }

    return 1;
}

int gpu_owner_run_repl(gpu_owner *owner, const char *fifo_path,
                       const char *log_path)
{
    FILE *in = stdin;
    FILE *log_file = NULL;
    char line[256] = {0};

    if (!owner)
        return -1;

    /* Unbuffered stdout so a hang names its line even without a newline flush. */
    setvbuf(stdout, NULL, _IONBF, 0);

    if (log_path) {
        log_file = fopen(log_path, "a");
        if (!log_file) {
            owner_log(owner, "run_repl: could not open log '%s' — using stdout",
                      log_path);
        } else {
            setvbuf(log_file, NULL, _IONBF, 0);
            owner->log = log_file;
        }
    }

    if (fifo_path) {
        in = fopen(fifo_path, "r");
        if (!in) {
            owner_log(owner, "run_repl: could not open fifo '%s'", fifo_path);
            if (log_file)
                fclose(log_file);
            return -1;
        }
    }

    owner_log(owner, "owner REPL ready (fifo=%s log=%s) — type 'help'",
              fifo_path ? fifo_path : "stdin", log_path ? log_path : "stdout");

    while (fgets(line, sizeof(line), in)) {
        if (!dispatch_command(owner, line))
            break;
    }

    owner_log(owner, "owner REPL exiting");
    if (fifo_path && in)
        fclose(in);
    if (log_file) {
        owner->log = stdout;
        fclose(log_file);
    }

    return 0;
}
