/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_claim.h
 * @brief Multi-tenant ownership of card resources — TLB windows and Tensix
 *   cores — for the userspace driver. This is the arbitration tt-kmd performs
 *   for us today, and it is the ONE part of the driver that must exist before
 *   the module can be removed, because losing it does not degrade performance,
 *   it corrupts other jobs.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A REQUIREMENT AND NOT A FEATURE
 * ---------------------------------------------------------------------------
 * The card is meant to host SEVERAL JOBS AT ONCE, each owning a set of cores,
 * so kernel sweeps can run in parallel instead of serialising behind one job.
 * tt-kmd supplies exactly one thing that makes that safe today: `ALLOCATE_TLB`
 * is a per-device bitmap shared across every open fd, so two
 * processes cannot claim the same one of the 202 windows. Nothing else in the
 * six ioctls we use is arbitration. Delete the module without replacing that
 * bitmap and two jobs silently share a window: the second CONFIGURE_TLB
 * re-aims it under the first, and the first's next store lands on the second's
 * endpoint. There is no error, no signal, and the symptom is a wrong answer or
 * a hung NoC in a process that did nothing wrong.
 *
 * ---------------------------------------------------------------------------
 * MECHANISM, AND WHY flock AND NOT A SHARED BITMAP
 * ---------------------------------------------------------------------------
 * One lock FILE per resource, in a per-namespace directory, claimed with
 * `flock(LOCK_EX | LOCK_NB)`:
 *
 *     /tmp/ft_tt_udrv.d/<card>/<namespace>/<index>.lock
 *
 * Three candidates were considered and this one is chosen for one decisive
 * reason: **the claim IS the lock, so there is no window in which two
 * processes both believe they own the resource, and no claim can outlive its
 * owner.**
 *
 *   - A SHARED MMAP'd BITMAP with a mutex is what the kernel does, and it is
 *     the wrong shape in userspace. Test-and-set is fine; the problem is
 *     death. A process killed with SIGKILL while holding bits leaves them set
 *     forever, so the design needs a liveness sweep — and a sweep keyed on pid
 *     is wrong across pid reuse, needs a start-time tiebreak, and is itself
 *     racy against a process that is claiming right now. Every one of those is
 *     a bug we would have to find on hardware.
 *   - A BROKER PROCESS is the strongest option (it could also serialise the
 *     register writes and own core-set policy) and it is rejected on the
 *     product goal: Phase 12 exists so a user plugs the card in and runs
 *     Pantheon. A daemon is another thing to install, supervise and version —
 *     it reintroduces the class of problem the kernel module is being removed
 *     for. Revisit only if a policy question appears that cannot be answered
 *     by "first claimant wins".
 *   - flock: the kernel releases every flock a process holds the instant its
 *     last descriptor for the file closes, for ANY reason including SIGKILL
 *     and an uncaught fault. Reclaim after abnormal death is therefore not
 *     code we write, it is a property we inherit. That is the same reasoning
 *     `tools/ft_tt_card_lock.c` already made in this tree for the whole-card
 *     lock, and this module is deliberately its sibling: same directory
 *     family, same claims-record convention, same staleness sweep.
 *
 * RACE BEHAVIOUR, stated precisely. Two processes calling
 * `ft_tt_udrv_claim_acquire()` for the same index at the same instant both
 * open the file and both call `flock(LOCK_EX|LOCK_NB)`. The kernel grants one
 * and returns `EWOULDBLOCK` to the other, which becomes FT_TT_UDRV_EAGAIN.
 * There is no torn state to clean up, because nothing is written before the
 * lock is held. The human-readable `.claim` record is written AFTER the lock
 * and is advisory reporting only — if a process dies between the two, the lock
 * is already free and the orphaned record is swept by pid liveness.
 *
 * ---------------------------------------------------------------------------
 * WHAT A RECLAIMED RESOURCE IS AND IS NOT
 * ---------------------------------------------------------------------------
 * A window handed back after an abnormal death is **not blank** — it is still
 * aimed wherever the dead process left it, and if its index is below 32 it may
 * still hold strided-multicast configuration. That is not a defect of this
 * module: tt-kmd has exactly the same property (it clears window
 * configuration on neither allocate nor free; 2 of 202 held live config when
 * the pool was last inspected — DEV_NOTES §1). The rule is unchanged and it is
 * the reason no zeroing pass is needed here: **AIM BEFORE YOU TOUCH.** A
 * claim grants the right to program a window, never the assumption that it is
 * unprogrammed.
 *
 * ---------------------------------------------------------------------------
 * COEXISTENCE WITH tt-kmd DURING THE TRANSITION
 * ---------------------------------------------------------------------------
 * While tt-kmd is bound it owns window 201 (`KERNEL_TLB_INDEX`) and uses it
 * for its own NoC reads — telemetry, ARC messaging, cable-fault checks. It
 * takes that window with `set_bit()` at probe and never publishes the fact
 * anywhere this module can see. So a reserved set is a parameter, not a
 * constant: pass `{201}` while the module is loaded, and the empty set once it
 * is not. Claiming 201 out from under a loaded tt-kmd would corrupt the
 * driver's own telemetry reads, which is the most confusing possible failure.
 *
 * ---------------------------------------------------------------------------
 * TWO LIMITS THAT MUST NOT BE OVER-PROMISED
 * ---------------------------------------------------------------------------
 * 1. **Correctness work parallelises; timing work does not.** One ASIC, one
 *    clock, one power budget. Concurrent jobs contend for DRAM, for the NoC
 *    and for the board's power envelope, and they contaminate each other's
 *    numbers. This project already owns the equivalent scar on the K80, where
 *    two busy dies of one board drop to 398 MHz and any parallel benchmark is
 *    void. Functional isolation is not timing isolation, and a core set does
 *    not make one job's measurement valid while another job runs.
 * 2. **The NoC is shared fabric, so isolation has a ceiling.** A bad multicast
 *    from one job reaches cores belonging to another — and that is precisely
 *    this project's most frequent hazard (more than 4 multicast destinations
 *    wedges reliably, fingerprint `0x00047000`). A core set isolates ordinary
 *    work; it does not contain that failure, and the recovery for it is
 *    whole-card (see `ft_tt_udrv.h` §5e).
 *
 * @status OFF-HARDWARE COMPLETE. This module touches no device and never has
 *   to: it is entirely filesystem coordination, so it is the one piece of the
 *   userspace driver that can be finished and gated before any bring-up.
 * @issues
 *   - flock is ADVISORY. A process that does not use this module can program
 *     any window it likes. This protects cooperating jobs, which is all
 *     tt-kmd's bitmap protected either — the bitmap only bound processes going
 *     through the ioctl.
 * @todo None
 */

#ifndef FT_TT_UDRV_CLAIM_H
#define FT_TT_UDRV_CLAIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/** Longest reason string a claim record keeps, including the terminator. */
#define FT_TT_UDRV_CLAIM_REASON_MAX 96u

/** Longest namespace name, including the terminator. */
#define FT_TT_UDRV_CLAIM_NAMESPACE_MAX 32u

/**
 * A claimable resource class. The namespace becomes a directory, so two
 * classes never collide even when their indices do — window 7 and core 7 are
 * different resources.
 */
typedef struct ft_tt_udrv_claim_namespace {
    char root[256];                              /**< coordination root, e.g. /tmp/ft_tt_udrv.d */
    char card[FT_TT_UDRV_CLAIM_NAMESPACE_MAX];   /**< which card, e.g. "0000:12:00.0" */
    char name[FT_TT_UDRV_CLAIM_NAMESPACE_MAX];   /**< "tlb2m" or "tensix" */
    uint32_t count;                              /**< number of indices, exclusive upper bound */
} ft_tt_udrv_claim_namespace;

/** One held claim. Opaque; freed by ft_tt_udrv_claim_release(). */
typedef struct ft_tt_udrv_claim ft_tt_udrv_claim;

/** What a scan found out about one index. */
typedef struct ft_tt_udrv_claim_info {
    uint32_t index;
    bool held;                                 /**< a live process holds the lock */
    pid_t pid;                                 /**< from the record; 0 if unknown */
    bool pid_alive;                            /**< kill(pid, 0) said yes */
    char reason[FT_TT_UDRV_CLAIM_REASON_MAX];  /**< from the record; "" if unknown */
} ft_tt_udrv_claim_info;

/**
 * Build a namespace descriptor. `root` may be NULL for the default
 * (`/tmp/ft_tt_udrv.d`); passing an explicit root is what lets a test drive
 * the whole module inside a scratch directory.
 *
 * @param out_namespace  receives the descriptor
 * @param root           coordination root, or NULL for the default
 * @param card           card identity, e.g. the PCI BDF. Must be non-empty and
 *                       must not contain '/' — it becomes a directory name
 * @param name           resource class, same rules
 * @param count          number of indices in the class
 * @return               FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_claim_namespace_init(ft_tt_udrv_claim_namespace *out_namespace, const char *root,
                                    const char *card, const char *name, uint32_t count);

/**
 * Claim one index exclusively.
 *
 * The claim is the flock, so it is atomic against every other process using
 * this module and is released by the kernel if this process dies for any
 * reason. The descriptor is opened `FD_CLOEXEC`, deliberately: a claim that
 * survived `exec()` into an unrelated helper would be held by a process that
 * does not know it holds it.
 *
 * @param ns          namespace
 * @param index       which resource, < ns->count
 * @param reason      short human-readable purpose, recorded for reporting.
 *                    May be NULL
 * @param out_claim   receives the claim; release with ft_tt_udrv_claim_release()
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_EAGAIN when another live process
 *                    holds it; FT_TT_UDRV_EINVAL on a bad argument;
 *                    FT_TT_UDRV_EIO on a filesystem failure
 *
 * Thread-safe: yes across processes. WITHIN one process flock is per open file
 *   description, so two threads that each open the file get independent
 *   descriptions and the second is refused — which is the behaviour you want.
 * Reentrant: no (allocates)   Signal-safe: no
 */
int ft_tt_udrv_claim_acquire(const ft_tt_udrv_claim_namespace *ns, uint32_t index,
                             const char *reason, ft_tt_udrv_claim **out_claim);

/**
 * Claim the first free index in `[0, ns->count)`, skipping `reserved`.
 *
 * This is the allocator `ALLOCATE_TLB` was, with the reserved set standing in
 * for the windows another driver owns (window 201 while tt-kmd is loaded).
 *
 * @param reserved       indices never to hand out; may be NULL if count is 0
 * @param reserved_count how many
 * @return               FT_TT_UDRV_OK; FT_TT_UDRV_EAGAIN when every index is
 *                       held or reserved (the `-ENOMEM` the ioctl returned);
 *                       FT_TT_UDRV_EINVAL; FT_TT_UDRV_EIO
 */
int ft_tt_udrv_claim_acquire_any(const ft_tt_udrv_claim_namespace *ns, const uint32_t *reserved,
                                 size_t reserved_count, const char *reason,
                                 ft_tt_udrv_claim **out_claim);

/** Which index a claim holds, or UINT32_MAX for NULL. */
uint32_t ft_tt_udrv_claim_index(const ft_tt_udrv_claim *claim);

/**
 * Drop the flock, remove the claim record, free the handle. NULL is a no-op.
 *
 * @return FT_TT_UDRV_OK, or FT_TT_UDRV_EIO (logged). The lock is released
 *         either way — the kernel does that when the descriptor closes.
 *
 * Thread-safe: no (frees `claim`)   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_claim_release(ft_tt_udrv_claim *claim);

/**
 * Report the state of every index, deleting orphaned records as it goes.
 *
 * "Held" is determined by attempting the lock, not by reading a record — a
 * record can lie, the lock cannot. Records exist only to name the holder.
 *
 * @param out_info   receives ns->count entries
 * @param info_cap   capacity of `out_info`; must be >= ns->count
 * @param out_held   receives how many indices are held. May be NULL
 * @return           FT_TT_UDRV_OK, FT_TT_UDRV_EINVAL, or FT_TT_UDRV_EIO
 */
int ft_tt_udrv_claim_scan(const ft_tt_udrv_claim_namespace *ns, ft_tt_udrv_claim_info *out_info,
                          size_t info_cap, uint32_t *out_held);

#endif /* FT_TT_UDRV_CLAIM_H */
