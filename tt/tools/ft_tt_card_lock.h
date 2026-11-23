/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_card_lock.h
 * @brief Advisory coordination for whole-card operations: an exclusive lock a card-level
 *   recovery rung (3, 4, 5) must hold before anything is reset, and a shared "claim" a
 *   long-running card user can register so a reset knows they exist and can name them.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS
 * ---------------------------------------------------------------------------
 * Rungs 3-5 reset the WHOLE CARD: every Tensix returns to power-on default. The reset
 * itself is cheap (~30 s) and reliable; what makes it expensive is that nothing coordinates
 * it, so a reset started by one agent destroys every other agent's in-flight work on the
 * same card (2026-08-12: four agents working the card concurrently, one card-level reset
 * would have wiped all four). This is coordination, not new recovery behaviour -- it does
 * not change what any rung does, and it is advisory: nothing stops a process that skips
 * this module entirely from resetting the card anyway. It only protects processes that go
 * through `ft_tt_core`, which is every sanctioned way to reset this card.
 *
 * ---------------------------------------------------------------------------
 * MECHANISM
 * ---------------------------------------------------------------------------
 * One `flock(2)` reader/writer lock on a single well-known file
 * (`/tmp/ft_tt_core_card.lock`): a shared claim takes `LOCK_SH`, an exclusive card-level
 * operation takes `LOCK_EX | LOCK_NB`. The kernel enforces the actual exclusion -- any
 * number of `LOCK_SH` holders may coexist, and `LOCK_EX` succeeds only when none do. This
 * is why flock is the right primitive here rather than a bare PID file: **the lock cannot
 * wedge on a dead holder.** The kernel releases every flock a process held the instant its
 * last fd for the file closes, for ANY reason -- normal exit, an uncaught signal, even
 * `SIGKILL`. The actual mutual exclusion is therefore self-healing and this module adds
 * nothing on top of it.
 *
 * What flock does NOT give you is who is holding the lock, or why. For that, every holder
 * -- shared claim or exclusive resetter -- writes a small text record naming itself
 * (`/tmp/ft_tt_core_card.claims/<pid>.claim`: pid, exclusive-or-shared, reason,
 * acquired-at) right after taking its flock, and removes it on a clean release. On a crash
 * the record is left behind as an ORPHAN: harmless to the real lock (already released by
 * the kernel) but misleading in a report, so every scan of the claims directory checks
 * each record's pid with `kill(pid, 0)` and deletes the ones that answer `ESRCH` before
 * reporting the rest. That is the stale-lock detection this module promises: detectable (a
 * dead pid), breakable (deleted automatically, no flag needed, no manual cleanup).
 *
 * `/tmp` is deliberate: this is single-host coordination -- a pid is only meaningful on the
 * machine that assigned it -- and `/tmp` being cleared across a reboot is desirable here,
 * since a reboot also invalidates every pid the claims directory could name.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS NOT
 * ---------------------------------------------------------------------------
 * Not a lock on any core. Rungs 0-2 touch one core each and never call into this module --
 * they are safe alongside other work on other cores by construction, and requiring this
 * lock for them would make the tool annoying for no safety gained. `status`, `probe` and
 * `topology` never call into this module either.
 *
 * @status New 2026-08-12, written for the four-concurrent-agents incident
 *   (`docs/tt_bringup_log.md` §21, `foundry_blackhole_backend_dev_notes.md` §8). Exercised
 *   via the refusal, acquisition, staleness and override paths -- a real rung 3/4/5 reset
 *   was deliberately never run against this build, since two other agents were live on the
 *   card at the time this was written.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CARD_LOCK_H
#define FT_TT_CARD_LOCK_H

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>
#include <time.h>

/** Longest free-text reason a holder may record, including the terminator. */
#define FT_TT_CARD_LOCK_REASON_MAX 160u

/** Most simultaneous holders a refusal will name. Generous for a handful of agents;
 *  contention beyond this still refuses correctly, it just stops naming names past it. */
#define FT_TT_CARD_LOCK_MAX_HOLDERS 32u

/** One line of "who is holding this, and why." */
typedef struct ft_tt_card_lock_holder {
    pid_t pid;
    bool exclusive;                       /**< true = a card-level op, false = a claim */
    char reason[FT_TT_CARD_LOCK_REASON_MAX];
    time_t acquired_at;
} ft_tt_card_lock_holder;

/** The live holders found on a refused or inspected lock, and how many stale (dead-pid)
 *  records this scan cleaned up along the way. */
typedef struct ft_tt_card_lock_status {
    unsigned holder_count;               /**< may exceed the array below; still counted */
    ft_tt_card_lock_holder holders[FT_TT_CARD_LOCK_MAX_HOLDERS];
    unsigned stale_cleaned;
} ft_tt_card_lock_status;

/**
 * Register a shared claim: "I am using this card, do not reset it out from under me."
 * Blocks until the shared lock is granted -- it never refuses outright, since any number
 * of shared claims may coexist; it only ever waits behind an exclusive card-level operation
 * already in progress.
 *
 * Touches no card state. Intended to be held for the lifetime of a long-running caller
 * (e.g. a foreground `ft_tt_core claim` left running in a terminal) and released with
 * ft_tt_card_lock_release() when that work is done.
 *
 * @param reason   free text, e.g. "probing cores 130-139"; NULL becomes "(no reason
 *                 given)"; truncated to FT_TT_CARD_LOCK_REASON_MAX - 1 bytes; embedded
 *                 newlines are flattened to spaces so the on-disk record stays line-based
 * @param out_fd   the open, locked fd; pass to ft_tt_card_lock_release() to give it up.
 *                 Left at -1 on any non-FT_TT_OK return.
 * @return         FT_TT_OK, or FT_TT_EIO if the lock file or claims directory could not be
 *                 opened/created
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_card_lock_claim_acquire(const char *reason, int *out_fd);

/**
 * Try to take the exclusive card-level lock. Never blocks.
 *
 * On success, `*out_fd` holds the lock; release it with ft_tt_card_lock_release() once the
 * card-level operation is complete or has failed -- EVERY exit path, including the failure
 * ones. On contention, FT_TT_EPERM is returned, `*out_fd` is left at -1, and `*out_status`
 * (if non-NULL) is filled with the live holders found. Either way this call also cleans up
 * any stale (dead-pid) claim records it finds along the way.
 *
 * @param reason      free text recorded against THIS process, so a caller who is in turn
 *                    someone else's obstacle can also be named; NULL becomes "(no reason
 *                    given)"
 * @param out_fd      -1 on any non-FT_TT_OK return
 * @param out_status  zeroed and filled; NULL is allowed if the caller does not need it
 * @return            FT_TT_OK, FT_TT_EPERM (another holder has it), or FT_TT_EIO
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_card_lock_exclusive_try(const char *reason, int *out_fd,
                                  ft_tt_card_lock_status *out_status);

/**
 * Look at who currently holds the lock without trying to acquire anything. Used to build
 * the "here is what --override-card-lock is about to override" banner.
 *
 * @return  FT_TT_OK once the scan ran (an empty status is a legitimate answer: nobody
 *          holds it), or FT_TT_EIO if the claims directory exists but could not be read
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_card_lock_inspect(ft_tt_card_lock_status *out_status);

/** Release a lock taken by either acquire function above: unlocks, closes `fd`, and
 *  removes this process's own claim record. Safe to call with fd == -1 (a no-op) -- so it
 *  is safe to call unconditionally on every exit path, including ones where the lock was
 *  never acquired (e.g. after --override-card-lock proceeded without it). */
void ft_tt_card_lock_release(int fd);

/** Human-readable account of who holds the lock, one line per holder, plus a note of any
 *  stale records this scan cleared. */
void ft_tt_card_lock_status_print(const ft_tt_card_lock_status *status);

#endif /* FT_TT_CARD_LOCK_H */
