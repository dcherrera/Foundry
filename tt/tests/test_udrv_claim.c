/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_claim.c
 * @brief Gate for ft_tt_udrv_claim.c — the multi-tenant ownership that
 *   replaces tt-kmd's ALLOCATE_TLB bitmap.
 *
 *   The claim mechanism's whole promise is that it survives an abnormal death,
 *   so the load-bearing stage here FORKS A CHILD THAT CLAIMS AND IS THEN
 *   KILLED. A suite that only ever tests acquire/release in one process
 *   verifies the easy half and misses the reason flock was chosen over a
 *   bitmap.
 *
 *   Everything runs inside a scratch root, so this never touches the real
 *   coordination directory and can run alongside live jobs.
 *
 * @status Off-hardware. Filesystem only; no device is opened.
 * @issues None
 * @todo None
 */

/* flock(2) is BSD, not POSIX, so a bare _POSIX_C_SOURCE hides it on both
 * platforms this code is built on. Same shape as the _DEFAULT_SOURCE note in
 * DEV_NOTES §5 for MADV_DONTNEED: macOS declares BSD interfaces under
 * _DARWIN_C_SOURCE, glibc under _DEFAULT_SOURCE (which implies POSIX 2008). */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _DEFAULT_SOURCE
#endif

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_claim.h"

#define TEST_WINDOW_COUNT 8u

static char scratch_root[192];

static void build_scratch_root(void)
{
    (void)snprintf(scratch_root, sizeof(scratch_root), "/tmp/ft_tt_udrv_claim_test.%ld",
                   (long)getpid());
}

static void remove_scratch_root(void)
{
    char command[256];

    (void)snprintf(command, sizeof(command), "rm -rf '%s'", scratch_root);
    if (system(command) != 0)
        (void)fprintf(stderr, "fixture: cleanup of %s failed\n", scratch_root);
}

static void stage_namespace_validation(void)
{
    ft_tt_udrv_claim_namespace ns;

    ft_tt_test_step("Stage 1 — a namespace becomes a directory, so it is validated like one");

    ft_tt_test_check("a well-formed namespace is accepted",
                     ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "0000:12:00.0", "tlb2m",
                                                     TEST_WINDOW_COUNT) == FT_TT_UDRV_OK,
                     "");

    ft_tt_test_check("a card name containing '/' is refused",
                     ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "../etc", "tlb2m",
                                                     8u) == FT_TT_UDRV_EINVAL,
                     "it would escape the root");

    ft_tt_test_check("a card name of \"..\" is refused",
                     ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "..", "tlb2m", 8u) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check("an empty resource class is refused",
                     ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "card", "", 8u) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check("a zero-sized namespace is refused",
                     ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "card", "tlb2m", 0u) ==
                         FT_TT_UDRV_EINVAL,
                     "nothing to claim");
}

static void stage_basic_ownership(void)
{
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim *first = NULL;
    ft_tt_udrv_claim *second = NULL;

    ft_tt_test_step("Stage 2 — one owner at a time");

    (void)ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "0000:12:00.0", "tlb2m",
                                          TEST_WINDOW_COUNT);

    ft_tt_test_check("window 3 can be claimed",
                     ft_tt_udrv_claim_acquire(&ns, 3u, "encoder gate", &first) == FT_TT_UDRV_OK,
                     "");
    ft_tt_test_check_fmt("the claim knows its index", ft_tt_udrv_claim_index(first) == 3u, "%u",
                         ft_tt_udrv_claim_index(first));

    /* A second independent open file description in the SAME process is
     * exactly what a second process looks like to flock, which is why this is
     * a meaningful check rather than a self-comparison. */
    ft_tt_test_check("a second claim on window 3 is refused",
                     ft_tt_udrv_claim_acquire(&ns, 3u, "should fail", &second) ==
                         FT_TT_UDRV_EAGAIN,
                     "already held");

    ft_tt_test_check("a different window is unaffected",
                     ft_tt_udrv_claim_acquire(&ns, 4u, "neighbour", &second) == FT_TT_UDRV_OK,
                     "");

    ft_tt_test_check("an out-of-range index is refused",
                     ft_tt_udrv_claim_acquire(&ns, TEST_WINDOW_COUNT, "oob", &second) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    (void)ft_tt_udrv_claim_release(second);
    ft_tt_test_check("releasing gives the window back",
                     ft_tt_udrv_claim_release(first) == FT_TT_UDRV_OK, "");

    ft_tt_test_check("...and it can then be re-claimed",
                     ft_tt_udrv_claim_acquire(&ns, 3u, "again", &first) == FT_TT_UDRV_OK, "");
    (void)ft_tt_udrv_claim_release(first);

    ft_tt_test_check("releasing NULL is a no-op",
                     ft_tt_udrv_claim_release(NULL) == FT_TT_UDRV_OK, "");
}

static void stage_allocate_any(void)
{
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim *held[TEST_WINDOW_COUNT];
    ft_tt_udrv_claim *extra = NULL;
    const uint32_t reserved[] = {0u, 7u};
    uint32_t index;
    uint32_t count = 0u;
    bool saw_reserved = false;

    ft_tt_test_step("Stage 3 — allocate-any, with the reserved set tt-kmd's window 201 needs");

    (void)ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "0000:12:00.0", "anytest",
                                          TEST_WINDOW_COUNT);

    for (index = 0u; index < TEST_WINDOW_COUNT; index++)
        held[index] = NULL;

    for (index = 0u; index < TEST_WINDOW_COUNT; index++) {
        if (ft_tt_udrv_claim_acquire_any(&ns, reserved, 2u, "sweep", &held[count]) !=
            FT_TT_UDRV_OK)
            break;

        if (ft_tt_udrv_claim_index(held[count]) == 0u ||
            ft_tt_udrv_claim_index(held[count]) == 7u)
            saw_reserved = true;

        count++;
    }

    ft_tt_test_check_fmt("6 of 8 windows are allocatable with 2 reserved", count == 6u, "%u",
                         count);
    ft_tt_test_check("no reserved window was ever handed out", !saw_reserved,
                     "0 and 7 skipped");
    ft_tt_test_check("exhaustion reports EAGAIN, which is the ioctl's -ENOMEM",
                     ft_tt_udrv_claim_acquire_any(&ns, reserved, 2u, "one too many", &extra) ==
                         FT_TT_UDRV_EAGAIN,
                     "");

    for (index = 0u; index < count; index++)
        (void)ft_tt_udrv_claim_release(held[index]);

    ft_tt_test_check("with nothing reserved, window 0 is available again",
                     ft_tt_udrv_claim_acquire_any(&ns, NULL, 0u, "unreserved", &extra) ==
                         FT_TT_UDRV_OK,
                     "");
    ft_tt_test_check_fmt("...and it is the lowest index", ft_tt_udrv_claim_index(extra) == 0u,
                         "%u", ft_tt_udrv_claim_index(extra));
    (void)ft_tt_udrv_claim_release(extra);
}

static void stage_death_reclaims(void)
{
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim *claim = NULL;
    ft_tt_udrv_claim_info info[TEST_WINDOW_COUNT];
    uint32_t held = 0u;
    int ready_pipe[2];
    char ready = 0;
    pid_t child;
    int status = 0;

    ft_tt_test_step("Stage 4 — THE LOAD-BEARING ONE: a killed holder gives the window back");

    (void)ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "0000:12:00.0", "death",
                                          TEST_WINDOW_COUNT);

    /* A pipe, not a poll loop. The first version of this stage had the parent
     * POLL by attempting the claim, which meant the parent could win the race
     * and the child then failed to acquire at all — the stage went red for a
     * reason that had nothing to do with the mechanism. A handshake removes
     * the race instead of widening the timeout. */
    if (pipe(ready_pipe) != 0) {
        ft_tt_test_check("pipe succeeded", false, "pipe failed");
        return;
    }

    child = fork();
    if (child == 0) {
        ft_tt_udrv_claim *child_claim = NULL;
        char signal_byte = 1;

        (void)close(ready_pipe[0]);

        if (ft_tt_udrv_claim_acquire(&ns, 2u, "child that dies", &child_claim) != FT_TT_UDRV_OK)
            _exit(3);

        /* Say the claim is held, then wait to be killed WITHOUT releasing
         * anything. SIGKILL is the case that matters: no cleanup path of ours
         * can run at all, so anything that gets the window back is the
         * kernel's doing. */
        if (write(ready_pipe[1], &signal_byte, 1u) != 1)
            _exit(4);

        for (;;)
            pause();
    }

    if (child < 0) {
        ft_tt_test_check("fork succeeded", false, "fork failed");
        return;
    }

    (void)close(ready_pipe[1]);
    ft_tt_test_check("the child says it holds window 2",
                     read(ready_pipe[0], &ready, 1u) == 1 && ready == 1, "handshake");
    (void)close(ready_pipe[0]);

    ft_tt_test_check("...and the parent is refused while it lives",
                     ft_tt_udrv_claim_acquire(&ns, 2u, "parent probe", &claim) ==
                         FT_TT_UDRV_EAGAIN,
                     "held by another process");

    (void)ft_tt_udrv_claim_scan(&ns, info, TEST_WINDOW_COUNT, &held);
    ft_tt_test_check_fmt("a scan sees exactly one window held", held == 1u, "%u", held);
    ft_tt_test_check_fmt("...and names the child as its holder", info[2].pid == child,
                         "pid %ld vs child %ld", (long)info[2].pid, (long)child);
    ft_tt_test_check_fmt("...with the reason it registered",
                         strcmp(info[2].reason, "child that dies") == 0, "%s", info[2].reason);

    (void)kill(child, SIGKILL);
    (void)waitpid(child, &status, 0);

    /* Scan BEFORE re-acquiring. The first version of this stage scanned after,
     * by which point release() had already unlinked the record and the sweep
     * had nothing left to do — so a build with the sweep DELETED passed. The
     * order is the check. */
    (void)ft_tt_udrv_claim_scan(&ns, info, TEST_WINDOW_COUNT, &held);
    ft_tt_test_check_fmt("nothing is held once the holder is dead", held == 0u, "%u", held);
    ft_tt_test_check_fmt("the orphaned record was swept by the scan", info[2].pid == 0,
                         "pid %ld", (long)info[2].pid);

    ft_tt_test_check("SIGKILL released the claim — the kernel does it, not our code",
                     ft_tt_udrv_claim_acquire(&ns, 2u, "after the death", &claim) ==
                         FT_TT_UDRV_OK,
                     "");
    (void)ft_tt_udrv_claim_release(claim);
}

static void stage_scan_reporting(void)
{
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim *claim = NULL;
    ft_tt_udrv_claim_info info[TEST_WINDOW_COUNT];
    uint32_t held = 99u;

    ft_tt_test_step("Stage 5 — the scan reports the LOCK, not the record");

    (void)ft_tt_udrv_claim_namespace_init(&ns, scratch_root, "0000:12:00.0", "scan",
                                          TEST_WINDOW_COUNT);

    (void)ft_tt_udrv_claim_scan(&ns, info, TEST_WINDOW_COUNT, &held);
    ft_tt_test_check_fmt("a fresh namespace has nothing held", held == 0u, "%u", held);

    (void)ft_tt_udrv_claim_acquire(&ns, 5u, "scan subject", &claim);
    (void)ft_tt_udrv_claim_scan(&ns, info, TEST_WINDOW_COUNT, &held);

    ft_tt_test_check_fmt("one window held", held == 1u, "%u", held);
    ft_tt_test_check("...and it is the right one", info[5].held && !info[4].held, "index 5");
    ft_tt_test_check_fmt("...owned by this process", info[5].pid == getpid() && info[5].pid_alive,
                         "pid %ld", (long)info[5].pid);

    ft_tt_test_check("a scan buffer that is too small is refused",
                     ft_tt_udrv_claim_scan(&ns, info, 2u, &held) == FT_TT_UDRV_EINVAL, "");

    (void)ft_tt_udrv_claim_release(claim);
    (void)ft_tt_udrv_claim_scan(&ns, info, TEST_WINDOW_COUNT, &held);
    ft_tt_test_check_fmt("releasing clears both the lock and the record",
                         held == 0u && info[5].pid == 0, "held=%u pid=%ld", held,
                         (long)info[5].pid);

    /* A record that names a LIVE process on a window nobody holds. This is the
     * pid-reuse hazard the design rejects a bitmap over, and it is the only
     * case that separates "the scan reports the LOCK" from "the scan reports
     * the RECORD" — without it, a scan built entirely on records passes. */
    {
        char record_path[256];
        FILE *forged;

        (void)snprintf(record_path, sizeof(record_path), "%s/0000:12:00.0/scan/6.claim",
                       scratch_root);
        forged = fopen(record_path, "w");
        if (forged) {
            (void)fprintf(forged, "pid %ld\nreason forged\nacquired 0\n", (long)getpid());
            (void)fclose(forged);
        }

        (void)ft_tt_udrv_claim_scan(&ns, info, TEST_WINDOW_COUNT, &held);
        ft_tt_test_check_fmt("a record naming a LIVE pid does not make a free window held",
                             !info[6].held && held == 0u, "held=%u index6=%d", held,
                             info[6].held ? 1 : 0);
        ft_tt_test_check("...and it is NOT swept, because its pid is alive",
                         info[6].pid == getpid(), "sweeping it would delete a real claim");
    }
}

static void stage_namespaces_are_separate(void)
{
    ft_tt_udrv_claim_namespace windows;
    ft_tt_udrv_claim_namespace cores;
    ft_tt_udrv_claim *window_claim = NULL;
    ft_tt_udrv_claim *core_claim = NULL;

    ft_tt_test_step("Stage 6 — window 7 and core 7 are different resources");

    (void)ft_tt_udrv_claim_namespace_init(&windows, scratch_root, "0000:12:00.0", "tlb2m", 8u);
    (void)ft_tt_udrv_claim_namespace_init(&cores, scratch_root, "0000:12:00.0", "tensix", 8u);

    ft_tt_test_check("window 7 claims",
                     ft_tt_udrv_claim_acquire(&windows, 7u, "w", &window_claim) == FT_TT_UDRV_OK,
                     "");
    ft_tt_test_check("core 7 claims independently",
                     ft_tt_udrv_claim_acquire(&cores, 7u, "c", &core_claim) == FT_TT_UDRV_OK, "");

    (void)ft_tt_udrv_claim_release(window_claim);
    (void)ft_tt_udrv_claim_release(core_claim);

    /* And a different card is a different namespace again — two cards in one
     * box must not share a window pool. */
    (void)ft_tt_udrv_claim_namespace_init(&cores, scratch_root, "0000:03:00.0", "tlb2m", 8u);
    ft_tt_test_check("window 7 of a SECOND card claims independently",
                     ft_tt_udrv_claim_acquire(&windows, 7u, "w", &window_claim) ==
                             FT_TT_UDRV_OK &&
                         ft_tt_udrv_claim_acquire(&cores, 7u, "w2", &core_claim) == FT_TT_UDRV_OK,
                     "");

    (void)ft_tt_udrv_claim_release(window_claim);
    (void)ft_tt_udrv_claim_release(core_claim);
}

int main(void)
{
    build_scratch_root();

    stage_namespace_validation();
    stage_basic_ownership();
    stage_allocate_any();
    stage_death_reclaims();
    stage_scan_reporting();
    stage_namespaces_are_separate();

    remove_scratch_root();
    return ft_tt_test_finish();
}
